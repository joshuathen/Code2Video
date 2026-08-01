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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from shared storyboard
        title = "The Invisible Handshake (BLE Exchange)"
        lines = [
            "Phones 'Pixel' and 'Nexus' encounter each other in range.",
            "They exchange Ephemeral IDs using local Bluetooth signals.",
            "'Nexus' saves the received ID in a local diary."
        ]
        self.setup_layout(title, lines)
        
        # Color definitions matching storyboard
        GREY_COLOR = "#AAAAAA"
        CYAN_COLOR = "#00FFFF"
        WHITE_COLOR = "#FFFFFF"
        BLUE_GLOW = "#5555FF"
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#55FF55"

        # === Animation for Lecture Line 1 ===
        # Phones 'Pixel' and 'Nexus' encounter each other in range.
        self.play(self.lecture[0].animate.set_color(GREY_COLOR))
        
        pixel_phone = RoundedRectangle(height=1.0, width=0.6, corner_radius=0.1, color=GREY_COLOR)
        nexus_phone = RoundedRectangle(height=1.0, width=0.6, corner_radius=0.1, color=GREY_COLOR)
        
        pixel_label = Text("Pixel", font_size=18, color=GREY_COLOR)
        nexus_label = Text("Nexus", font_size=18, color=GREY_COLOR)
        
        # Positioning primary visual elements in columns 2 and 5 (L010)
        self.place_at_grid(pixel_phone, "C2")
        self.place_at_grid(nexus_phone, "C5")
        # Labels within 1 grid unit (L003)
        self.place_at_grid(pixel_label, "B2") 
        self.place_at_grid(nexus_label, "B5")
        
        # Proximity glow (Animation 2)
        # Using area between phones (C2 to C5) - L017 distribute visuals
        glow = Rectangle(height=1.5, width=2.5, color=BLUE_GLOW, fill_opacity=0.1, stroke_width=0)
        self.place_in_area(glow, "B3", "D4")

        # Bluetooth ripples (Animation 1)
        ripples = VGroup(*[
            Circle(radius=0.1, color=CYAN_COLOR, stroke_width=2).move_to(self.grid["C2"])
            for _ in range(3)
        ])
        
        self.play(
            Create(pixel_phone), Create(nexus_phone),
            Write(pixel_label), Write(nexus_label),
            FadeIn(glow)
        )
        
        # Show encounter via ripples
        self.play(
            AnimationGroup(
                *[ripple.animate(run_time=1.5, rate_func=linear).scale(10).set_stroke(opacity=0) for ripple in ripples],
                lag_ratio=0.3
            )
        )
        self.remove(ripples)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They exchange Ephemeral IDs using local Bluetooth signals.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CYAN_COLOR)
        )
        
        # Animation 3: Show central white server icon with a large red 'X' (L001: avoid Row A)
        server_icon = Square(side_length=0.6, color=WHITE_COLOR)
        server_label = Text("Server", font_size=14, color=WHITE_COLOR)
        server_group = VGroup(server_icon, server_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(server_group, "B3")
        
        cross = VGroup(
            Line(UL, DR, color=RED_COLOR, stroke_width=6),
            Line(UR, DL, color=RED_COLOR, stroke_width=6)
        ).scale(0.3)
        self.place_at_grid(cross, "B3")
        
        # Animation 5: Animate 'EphID' labels moving (Step 5)
        # Fix Issue 30: self.place_at_grid('ephid_text', 'C2', scale_factor=0.6)
        ephid_text = Text("EphID_88", font_size=20, color=CYAN_COLOR)
        self.place_at_grid(ephid_text, "C2", scale_factor=0.6)
        
        self.play(FadeIn(server_group))
        self.play(Create(cross))
        self.play(ephid_text.animate(run_time=2).move_to(self.grid["C5"]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'Nexus' saves the received ID in a local diary.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE_COLOR)
        )
        
        # Animation 4: Display white 'Seen Diary' box (Step 6)
        # Fix Issue 32: self.place_at_grid('diary_box', 'D5', scale_factor=0.7)
        diary_box = Rectangle(height=0.8, width=1.4, color=WHITE_COLOR)
        self.place_at_grid(diary_box, "D5", scale_factor=0.7)
        
        # Fix Issue 31: self.place_in_area('diary_label', 'E5', 'F6', scale_factor=0.8)
        diary_label = Text("Seen Diary", font_size=20, color=WHITE_COLOR)
        self.place_in_area(diary_label, "E5", "F6", scale_factor=0.8)
        
        self.play(Create(diary_box), Write(diary_label))
        
        # Animation 5: Move EphID into diary and flash green (Step 7)
        self.play(
            ephid_text.animate.move_to(self.grid["D5"]).scale(0.8),
        )
        self.play(
            ephid_text.animate.set_color(GREEN_COLOR),
            diary_box.animate.set_stroke(color=GREEN_COLOR, width=5),
            run_time=0.5
        )
        self.play(
            diary_box.animate.set_stroke(color=WHITE_COLOR, width=2),
            run_time=0.5
        )
        self.wait(2)
