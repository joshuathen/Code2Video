from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        title_text = "Summary: The Verdict"
        lecture_lines = [
            "The verdict is in: the potato is a stem.",
            "Classification depends on structure, not just location.",
            "Case closed on this underground biological mystery!"
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Visual: Potato image represented by an Ellipse
        potato_1 = Ellipse(width=2.8, height=2.0, color="#D2B48C", fill_opacity=1.0)
        self.place_in_area(potato_1, "B2", "E5")
        
        # Visual: Large green checkmark (#00FF00)
        checkmark = VGroup(
            Line(LEFT * 0.4 + DOWN * 0.2, ORIGIN, color="#00FF00"),
            Line(ORIGIN, RIGHT * 0.8 + UP * 0.6, color="#00FF00")
        ).set_stroke(width=10)
        self.place_at_grid(checkmark, "C3", scale_factor=1.2)
        
        # Visual: 'STEM' in bold white text
        stem_label = Text("STEM", font_size=48, weight=BOLD, color=WHITE)
        self.place_at_grid(stem_label, "C4", scale_factor=1.0)
        
        self.play(FadeIn(potato_1))
        self.play(Create(checkmark), Write(stem_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line focus
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Clear previous visuals for the diagram
        self.play(FadeOut(potato_1), FadeOut(checkmark), FadeOut(stem_label))

        # Visual: Diagram of potato plant
        # Ground level line
        ground = Line(self.grid["C1"], self.grid["C6"], color="#8B4513", stroke_width=4)
        # Main stalk (above ground)
        stalk = Line(self.grid["A3"], self.grid["C3"], color="#228B22", stroke_width=6)
        # Tuber (underground)
        tuber_final = Ellipse(width=1.4, height=1.0, color="#D2B48C", fill_opacity=1.0)
        self.place_at_grid(tuber_final, "E5")
        # Underground stem connecting tuber to main stalk, highlighted (#32CD32)
        underground_stem = Line(self.grid["C3"], self.grid["E5"], color="#32CD32", stroke_width=12)
        
        self.play(Create(ground), Create(stalk))
        self.play(FadeIn(tuber_final))
        self.play(Create(underground_stem))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line focus
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Visual: 'CASE CLOSED' stamp (#FF0000)
        stamp_text = Text("CASE CLOSED", font_size=44, weight=BOLD, color="#FF0000")
        stamp_box = SurroundingRectangle(stamp_text, color="#FF0000", buff=0.2, stroke_width=8)
        stamp = VGroup(stamp_box, stamp_text).rotate(15 * DEGREES)
        # Position in center of visual area
        self.place_in_area(stamp, "B2", "E5", scale_factor=1.2)
        
        # Simulate a stamp action
        stamp.scale(1.5)
        self.play(FadeIn(stamp), stamp.animate.scale(1/1.5), run_time=0.5)
        self.wait(2)
