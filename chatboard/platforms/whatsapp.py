import aiohttp



class WhatsAppException(Exception):
    DEFAULT_ERROR_CODE_LOOKUP = {
        # General Graph API / OAuth Errors
        1: "An unknown error occurred.",
        2: "Service temporarily unavailable.",
        4: "Application request limit reached (throttling).",
        10: "Application does not have permission for this action.",
        17: "User request limit reached.",
        100: "Invalid parameter.",
        102: "Session key invalid or no longer valid.",
        190: "Invalid OAuth 2.0 Access Token.",

        # WhatsApp-Specific Errors - Recipient / Phone Number Issues
        131000: "Invalid recipient phone number.",
        131001: "Recipient phone number not subscribed to the application.",
        131002: "Message blocked by the user.",
        131009: "Recipient phone number is not associated with a WhatsApp account.",
        131010: "Recipient phone number format is incorrect.",
        131011: "Recipient phone number not in an allowed country.",
        131012: "Message blocked by WhatsApp (recipient cannot be messaged).",

        # WhatsApp-Specific Errors - Message and Media
        132000: "Invalid media type.",
        132001: "Unsupported message type.",
        132002: "Media upload failed.",
        132003: "Invalid message template or template parameters.",
        132005: "Interactive message content error.",
        132006: "Sticker message content error.",
        132007: "Template message content error.",

        # WhatsApp-Specific Errors - Business / Verification
        133000: "Business not verified.",
        133001: "Business restricted from sending messages.",

        # WhatsApp-Specific Errors - Message Sending & Templates
        135000: "Could not send the message.",
        135001: "Message template not found.",
        135002: "Invalid message template parameters.",
        136000: "One or more recipients are invalid or have not opted in.",

        # Rate Limiting
        80004: "Rate limit reached."
    }
    ERROR_CODE_LOOKUP = {}
    
    
    # def __init__(self, message, error_code: int, fb_trace_id: str | None = None, response=None):
    #     error_code_lookup = self.DEFAULT_ERROR_CODE_LOOKUP | self.ERROR_CODE_LOOKUP
    #     code_message = error_code_lookup.get(error_code, "Unknown error")
    #     self.message = message
    #     self.error_code = error_code
    #     self.fb_trace_id = fb_trace_id
    #     self.response = response
    #     self.code_message = code_message
    #     super().__init__(message)
    def __init__(self, message, error):
        error_code_lookup = self.DEFAULT_ERROR_CODE_LOOKUP | self.ERROR_CODE_LOOKUP
        error_code = error.get("code")
        fbtrace_id = error.get("fbtrace_id")
        details = error.get("details")
        code_message = error_code_lookup.get(error_code, "Unknown error")
        self.message = message
        self.error_code = error_code
        self.fbtrace_id = fbtrace_id
        self.details = details
        self.code_message = code_message
        super().__init__(message)
        
    def __str__(self) -> str:
        base_str = super().__str__()
        if self.error_code:
            base_str += f" [Error Code: {self.error_code}, {self.code_message}]"
        if self.details:
            base_str += f" [Details: {self.details}]"
        if self.fbtrace_id:
            base_str += f" [Trace ID: {self.fbtrace_id}]"
        return base_str
        
        
class WhatsAppSubscribedAppException(WhatsAppException):
    ERROR_CODE_LOOKUP = {
        100: "Invalid parameter",
        200: "Permissions error",
        2200: "subscription validation failed",
        2201: "received an invalid hub.challenge while validating endpoint",
    }





class FBGraphRequest:    

    API_URL = "https://graph.facebook.com"
    
    def __init__(self, access_token=None, version="v21.0"):
        self._access_token = access_token
        self._version = version
        
    def _build_url(self, url):
        fb_url = f"{self.API_URL}/{self._version}/" + url if not url.startswith("http") else url
        print(fb_url)
        return fb_url
    
    async def _unpack_error(self, response):
        error = await response.json()
        error = error.get("error", {})
        return error
    
    def _get_headers(self, add_authorization=True):
        headers = {
            "Content-Type": "application/json"
        }
        if add_authorization:
            if not self._access_token:
                raise ValueError("Access token is required")
            headers["Authorization"] = f"Bearer {self._access_token}"            
        return headers
    
    async def get(
            self, 
            url: str, 
            add_authorization: bool = True, 
            wa_exception_cls=WhatsAppException, 
            error_msg="Facebook graph Error"
        ):
        async with aiohttp.ClientSession() as session:
            headers = self._get_headers(add_authorization)
            async with session.get(
                self._build_url(url), 
                headers=headers
            ) as response:
                if response.status != 200:
                    error = await self._unpack_error(response)
                    # raise wa_exception_cls(error_msg, error.get("code"), error.get("fbtrace_id"), response=response)
                    raise wa_exception_cls(error_msg, error)
                j = await response.json()
                return j
    
    async def post(
            self, 
            url: str, 
            payload: dict | None = None, 
            add_authorization: bool = True,
            wa_exception_cls=WhatsAppException, 
            error_msg="Facebook graph Error"
        ):
        async with aiohttp.ClientSession() as session:
            headers = self._get_headers(add_authorization)
            async with session.post(
                self._build_url(url), 
                json=payload, 
                headers=headers
            ) as response:
                if response.status != 200:
                    error = await self._unpack_error(response)            
                    # raise wa_exception_cls(error_msg, error.get("code"), error.get("fbtrace_id"), response=response)
                    raise wa_exception_cls(error_msg, error)
                j = await response.json()
                return j








class WhatsAppClient:
    

    def __init__(self, access_token=None, version="v21.0") -> None:
        self._access_token = access_token
        self._version = version
        self.graph_request = FBGraphRequest(access_token, version)
        
        
    
    async def send_text_message(self, phone_number: str, message: str):
        """
            send a text message to a phone number.
            Attributes:
                phone_number: str - The recipient's phone number.
                message: str - The message to send.
        """
        payload = {
            "messaging_product": 'whatsapp',
            "to": phone_number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message
            }
        }
        return await self.graph_request.post("messages", payload, wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def code_to_access_token(self, code: str, app_id: str, app_secret: str):
        """
            get the access token from the code.
            Attributes:
                code: str - The code received from the WhatsApp callback.
                app_id: str - The customer's app ID.
                app_secret: str - The customer's app secret.
        """
        return await self.graph_request.get(f"oauth/access_token?client_id={app_id}&client_secret={app_secret}&code={code}", add_authorization=False)
    
    
    async def get_subscribed_apps(self, waba_id: str):
        """
            get the list of apps that are subscribed to the WhatsApp Business API client.
            Attributes:
                waba_id: str - The customer's WABA ID.
        """
        return await self.graph_request.get(f"{waba_id}/subscribed_apps", wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def subscribe_app(self, waba_id: str, override_callback_uri: str | None = None, verify_token: str | None = None):
        """
        subscribe an app to the WhatsApp Business API client.
        Attributes:            
        """
        payload = None
        if override_callback_uri:
            if not verify_token:
                raise ValueError("verify_token is required when override_callback_uri is provided")        
            payload = {
                "override_callback_uri": override_callback_uri,
                "verify_token": verify_token
            }
        return await self.graph_request.post(f"{waba_id}/subscribed_apps", payload, wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def delete_app_subscription(self, waba_id: str):
        return await self.graph_request.post(f"{waba_id}/subscribed_apps", wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def register_phone_number(self, phone_number_id: str, pin: str):
        """
            Attributes:
                phone_number_id: str - The customer's business phone number ID returned.
                pin: str - Set this value to a 6-digit number. This will be the business phone number's two-step verification PIN.
        """
        payload = {
            "messaging_product": "whatsapp",
            "pin": pin
        }
        return await self.graph_request.post(f"{phone_number_id}/register", payload, wa_exception_cls=WhatsAppSubscribedAppException)
    
    async def get_phone_number_info(self, waba_id: str):
        """
        get the list of phone numbers associated with the WhatsApp Business API client.
        """
        return await self.graph_request.get(f"{waba_id}/phone_numbers")
    
    async def deregister_phone_number(self, phone_number_id: str):
        """
            deregister a phone number from the WhatsApp Business API client.
            Attributes:
                phone_number_id: str - The customer's business phone number ID returned.
        """
        return await self.graph_request.post(f"{phone_number_id}/deregister", wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    
    
    
    
    
    
    
    
    
    
    async def get_app_subscriptions(self, waba_id: str):
        """
            get the list of pages that are subscribed to the app.
        """
        return await self.graph_request.get(f"{waba_id}/subscriptions", wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def subscribe_app_to_page(self, app_id: str, page_id: str, subscribed_fields: list[str]):
        payload = {
            "subscribed_fields": subscribed_fields
        }
        return await self.graph_request.post(f"{app_id}/subscriptions", payload, wa_exception_cls=WhatsAppSubscribedAppException)
    
    
    async def get_page_subscriptions(self, app_id: str, page_id: str):
        return await self.graph_request.get(f"{app_id}/subscriptions", wa_exception_cls=WhatsAppSubscribedAppException)
    
    