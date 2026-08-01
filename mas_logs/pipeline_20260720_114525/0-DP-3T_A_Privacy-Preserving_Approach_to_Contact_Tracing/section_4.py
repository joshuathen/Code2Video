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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "Core Mechanism: Bluetooth Low Energy (BLE) and Random IDs"
        lecture_lines = [
            "If diagnosed, a user reports their TRACE ID history.",
            "This list is uploaded to a central server.",
            "No personal information is shared.",
            "Only anonymized encounter data is uploaded.",
            "This enables tracking exposure without compromising privacy."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # If diagnosed, a user reports their TRACE ID history.
        trace_id_history = Text("TRACE ID History", font_size=24).move_to(self.grid["A1"])
        user_icon = Circle(radius=0.3, color=BLUE).move_to(self.grid["B1"])
        report_arrow = Arrow(user_icon.get_right(), trace_id_history.get_left(), buff=0.1)
        self.play(Write(trace_id_history), Create(user_icon), Create(report_arrow))
        self.wait(1)
        self.play(FadeOut(trace_id_history), FadeOut(user_icon), FadeOut(report_arrow))

        # === Animation for Lecture Line 2 ===
        # This list is uploaded to a central server.
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", fill_color=GRAY).scale(0.8).move_to(self.grid["A2"]) # Assuming server.svg exists
        upload_arrow = Arrow(self.grid["B2"], server_icon.get_center(), buff=0.1)
        self.play(Create(server_icon), Create(upload_arrow))
        self.wait(1)
        self.play(FadeOut(server_icon), FadeOut(upload_arrow))

        # === Animation for Lecture Line 3 ===
        # No personal information is shared.
        personal_info = Text("Personal Info", font_size=20, color=RED).move_to(self.grid["A3"])
        shared_icon = Text("X", font_size=40, color=RED).move_to(self.grid["B3"])
        self.play(Write(personal_info), Write(shared_icon))
        self.wait(1)
        self.play(FadeOut(personal_info), FadeOut(shared_icon))

        # === Animation for Lecture Line 4 ===
        # Only anonymized encounter data is uploaded.
        anonymized_data = Text("Anonymized Data", font_size=20, color=GREEN).move_to(self.grid["A4"])
        encounter_icon = Circle(radius=0.4, color=GREEN).move_to(self.grid["B4"])
        upload_arrow_anon = Arrow(encounter_icon.get_right(), anonymized_data.get_left(), buff=0.1)
        self.play(Write(anonymized_data), Create(encounter_icon), Create(upload_arrow_anon))
        self.wait(1)
        self.play(FadeOut(anonymized_data), FadeOut(encounter_icon), FadeOut(upload_arrow_anon))

        # === Animation for Lecture Line 5 ===
        # This enables tracking exposure without compromising privacy.
        tracking_icon = Dot(color=YELLOW).scale(2).move_to(self.grid["A5"])
        # The 'Shield' mobject is not a built-in Manim class.
        # We'll replace it with a common representation like a circle or a more complex shape if available.
        # For now, let's use a Circle as a placeholder for a shield-like shape.
        privacy_shield = Circle(radius=0.5, color=PURPLE, fill_opacity=0.5).move_to(self.grid["B5"])
        self.play(Create(tracking_icon), Create(privacy_shield))
        self.wait(2)
        self.play(FadeOut(tracking_icon), FadeOut(privacy_shield))