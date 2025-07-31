import * as React from "react"
import { cn } from "@/lib/utils"

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "info" | "destructive"
}

export const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ className, variant = "info", ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "rounded-md border p-3 text-sm",
          variant === "info" && "bg-blue-50 text-blue-700 border-blue-200",
          variant === "destructive" && "bg-red-50 text-red-700 border-red-200",
          className
        )}
        {...props}
      />
    )
  }
)
Alert.displayName = "Alert" 