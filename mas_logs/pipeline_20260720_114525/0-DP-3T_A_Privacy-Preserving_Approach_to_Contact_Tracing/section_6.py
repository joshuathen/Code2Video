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

# Grid layout (right side only):
# lecture |  A1  A2  A3  A4  A5  A6
#         |  B1  B2  B3  B4  B5  B6
#         |  C1  C2  C3  C4  C5  C6
#         |  D1  D2  D3  D4  D5  D6
#         |  E1  E2  E3  E4  E5  E6
#         |  F1  F2  F3  F4  F5  F6

class Section6Scene(TeachingScene):
    def construct(self):
        title = "How Exposure is Determined (and Privacy is Maintained)"
        lecture_lines = [
            "DP-3T prioritizes user privacy.",
            "No location or personal data is stored centrally.",
            "Sensitive data stays on the user's device.",
            "Temporary IDs prevent tracking and re-identification.",
            "It's a secure way to trace contacts."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # DP-3T prioritizes user privacy.
        privacy_text = Text("DP-3T", font_size=24).set_color(WHITE)
        self.play(FadeIn(privacy_text, shift=DOWN))
        self.place_at_grid(privacy_text, "A1")
        self.play(FadeOut(privacy_text, shift=UP))

        # === Animation for Lecture Line 2 ===
        # No location or personal data is stored centrally.
        central_server = Text("Central Server", font_size=24).set_color(WHITE)
        self.play(FadeIn(central_server, shift=DOWN))
        self.place_at_grid(central_server, "B2")
        self.play(FadeOut(central_server, shift=UP))

        # === Animation for Lecture Line 3 ===
        # Sensitive data stays on the user's device.
        device_data = Text("User Device", font_size=24).set_color(WHITE)
        self.play(FadeIn(device_data, shift=DOWN))
        self.place_at_grid(device_data, "C3")
        self.play(FadeOut(device_data, shift=UP))

        # === Animation for Lecture Line 4 ===
        # Temporary IDs prevent tracking and re-identification.
        temp_ids = Text("Temp IDs", font_size=24).set_color(WHITE)
        self.play(FadeIn(temp_ids, shift=DOWN))
        self.place_at_grid(temp_ids, "D4")
        self.play(FadeOut(temp_ids, shift=UP))

        # === Animation for Lecture Line 5 ===
        # It's a secure way to trace contacts.
        secure_trace = Text("Secure Trace", font_size=24).set_color(WHITE)
        self.play(FadeIn(secure_trace, shift=DOWN))
        self.place_at_grid(secure_trace, "E5")
        self.play(FadeOut(secure_trace, shift=UP))
