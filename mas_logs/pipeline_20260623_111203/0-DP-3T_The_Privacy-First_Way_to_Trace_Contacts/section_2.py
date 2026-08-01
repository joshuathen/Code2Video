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

class Section2Scene(TeachingScene):
    def construct(self):
        lines = [
            'A hash function creates a unique digital fingerprint.',
            'It is a one-way process, impossible to reverse.',
            'Tiny input changes produce completely different results.'
        ]
        self.setup_layout("Prerequisite: Cryptographic Hash Functions", lines)

        # Colors
        COLOR_INPUT = "#3498DB"
        COLOR_HASH = "#95A5A6"
        COLOR_FINGERPRINT = "#ECF0F1"
        COLOR_BARRIER = "#E74C3C"
        COLOR_DIFF = "#F1C40F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_INPUT)
        
        input_box = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=1.4, color=COLOR_INPUT),
            Text("Input Data", font_size=14, color=COLOR_INPUT)
        )
        hash_machine = VGroup(
            Rectangle(height=1.2, width=1.5, fill_opacity=0.3, color=COLOR_HASH),
            Text("Hash\nFunction", font_size=14, color=COLOR_HASH)
        )
        fingerprint = Text("0x7a3f...9e", font_size=16, color=COLOR_FINGERPRINT)

        self.place_at_grid(input_box, "B1")
        self.place_in_area(hash_machine, "B3", "C4")
        self.place_at_grid(fingerprint, "B6")

        self.play(FadeIn(input_box), Create(hash_machine))
        self.play(input_box.animate.move_to(self.grid["B3"]), run_time=1)
        self.play(FadeOut(input_box, shift=RIGHT*0.5), FadeIn(fingerprint, shift=RIGHT*0.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_BARRIER)
        
        arrow_back = Arrow(start=self.grid["B6"], end=self.grid["B4"], color=COLOR_BARRIER, buff=0.1)
        barrier_icon = VGroup(
            Circle(radius=0.25, color=COLOR_BARRIER, stroke_width=6),
            Line(start=[-0.15, 0.15, 0], end=[0.15, -0.15, 0], color=COLOR_BARRIER, stroke_width=6)
        ).move_to(self.grid["B4"])

        self.play(Create(arrow_back))
        self.play(Create(barrier_icon))
        self.play(Indicate(barrier_icon, color=COLOR_BARRIER))
        self.wait(1)
        
        self.play(FadeOut(arrow_back), FadeOut(barrier_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_DIFF)

        # Use MarkupText to show the tiny change
        input_box_new = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=1.4, color=COLOR_INPUT),
            MarkupText('Input Dat<span color="#E74C3C">b</span>', font_size=14, color=COLOR_INPUT)
        )
        fingerprint2 = Text("0x12b8...4c", font_size=16, color=COLOR_DIFF)

        self.place_at_grid(input_box_new, "D1")
        self.place_at_grid(fingerprint2, "D6")

        self.play(FadeIn(input_box_new))
        self.play(input_box_new.animate.move_to(self.grid["C3"]), run_time=1)
        self.play(FadeOut(input_box_new, shift=RIGHT*0.5), FadeIn(fingerprint2, shift=RIGHT*0.5))
        
        # Highlight difference
        self.play(
            Indicate(fingerprint, color=COLOR_DIFF),
            Indicate(fingerprint2, color=COLOR_DIFF)
        )
        self.wait(2)
