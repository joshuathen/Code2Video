from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Decentralized Data Storage and Anonymity"
        lecture_lines = [
            "The server checks reported IDs against known positive ones.",
            "If a match is found, an alert is sent.",
            "Individuals are notified of potential exposure.",
            "Their phones received a matched TRACE ID.",
            "This process protects user identity."
        ]
        self.setup_layout(title, lecture_lines)

        # Define colors for lecture lines
        colors = [
            "#FFD700",  # Gold
            "#FFA500",  # Orange
            "#FF6347",  # Tomato
            "#FF4500",  # OrangeRed
            "#DA70D6"   # Orchid
        ]

        # === Animation for Lecture Line 1 ===
        # The server checks reported IDs against known positive ones.
        server_text = Text("Server", color=colors[0]).scale(0.7)
        ids_text = Text("Reported IDs", color=colors[0]).scale(0.7)
        positive_ids_text = Text("Positive IDs", color=colors[0]).scale(0.7)

        self.place_at_grid(server_text, 'B2')
        self.place_at_grid(ids_text, 'C3')
        self.place_at_grid(positive_ids_text, 'C4')

        self.play(
            FadeIn(server_text),
            FadeIn(ids_text),
            FadeIn(positive_ids_text),
            self.lecture[0].animate.set_color(colors[0])
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # If a match is found, an alert is sent.
        alert_text = Text("ALERT!", color=colors[1]).scale(1.0)
        self.place_at_grid(alert_text, 'D3')

        self.play(
            FadeOut(ids_text),
            FadeOut(positive_ids_text),
            FadeIn(alert_text),
            self.lecture[1].animate.set_color(colors[1])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Individuals are notified of potential exposure.
        notification_text = Text("Notification", color=colors[2]).scale(0.7)
        exposure_text = Text("Potential Exposure", color=colors[2]).scale(0.7)

        self.place_at_grid(notification_text, 'E4')
        self.place_at_grid(exposure_text, 'F5')

        self.play(
            FadeOut(alert_text),
            FadeIn(notification_text),
            FadeIn(exposure_text),
            self.lecture[2].animate.set_color(colors[2])
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Their phones received a matched TRACE ID.
        phone_text = Text("Phone", color=colors[3]).scale(0.7)
        trace_id_text = Text("Matched TRACE ID", color=colors[3]).scale(0.7)

        self.place_at_grid(phone_text, 'D5')
        self.place_at_grid(trace_id_text, 'E6')

        self.play(
            FadeOut(notification_text),
            FadeOut(exposure_text),
            FadeIn(phone_text),
            FadeIn(trace_id_text),
            self.lecture[3].animate.set_color(colors[3])
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This process protects user identity.
        identity_protected_text = Text("Identity Protected", color=colors[4]).scale(0.8)
        # Assuming SVGIcon is not directly available, using a placeholder or Text for lock
        # If SVGIcon is intended to be used, ensure it's properly imported or replaced
        # For now, let's use Text as a fallback.
        lock_icon = Text("🔒", color=colors[4]).scale(0.8) # Using emoji as a placeholder for lock icon

        self.place_at_grid(identity_protected_text, 'B4')
        self.place_at_grid(lock_icon, 'A5')

        self.play(
            FadeOut(phone_text),
            FadeOut(trace_id_text),
            FadeIn(identity_protected_text),
            FadeIn(lock_icon),
            self.lecture[4].animate.set_color(colors[4])
        )
        self.wait(2)
