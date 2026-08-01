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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title_text = "Prerequisite: The Slope of a Straight Line"
        lecture_lines = [
            "Imagine climbing a ramp with a steady incline.",
            "The slope, rise over run, stays constant here.",
            "A straight line has the same rate everywhere."
        ]
        
        # Setup base layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors defined in requirements
        COLOR_RAMP = "#FFFFFF"
        COLOR_RISE = "#00FF00"
        COLOR_RUN = "#0000FF"
        COLOR_CHAR = YELLOW # Highlighting character and active line 3
        
        # === Animation for Lecture Line 1 ===
        # "Imagine climbing a ramp with a steady incline."
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Define the ramp. 30-degree incline.
        angle = 30 * DEGREES
        ramp_line = Line(LEFT * 2.5, RIGHT * 2.5, color=COLOR_RAMP).rotate(angle)
        
        # Issue 32: properly anchored to the right-side visual grid
        self.place_in_area(ramp_line, 'A2', 'F6', scale_factor=0.9)
        
        # Character represented as a small triangle
        character = Triangle(color=COLOR_CHAR, fill_opacity=1).scale(0.15)
        character.rotate(angle - 90 * DEGREES) # Align with ramp incline
        character.move_to(ramp_line.get_start())
        
        self.play(Create(ramp_line))
        self.play(FadeIn(character))
        
        # Small movement up the ramp
        self.play(character.animate.move_to(ramp_line.point_from_proportion(0.2)), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "The slope, rise over run, stays constant here."
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_RISE)
        )
        
        # Construct the Rise and Run triangle
        # We'll build it starting from the character's current position
        curr_char_pos = character.get_center()
        tri_run_val = 1.0
        tri_rise_val = tri_run_val * np.tan(angle)
        
        line_run = Line(curr_char_pos, curr_char_pos + [tri_run_val, 0, 0], color=COLOR_RUN)
        line_rise = Line(curr_char_pos + [tri_run_val, 0, 0], curr_char_pos + [tri_run_val, tri_rise_val, 0], color=COLOR_RISE)
        
        # Labels - Issues 33 and 34: Positioned using grid anchors
        run_text = Text("Run", font_size=20, color=COLOR_RUN)
        rise_text = Text("Rise", font_size=20, color=COLOR_RISE)
        
        self.place_at_grid(run_text, 'E3', scale_factor=0.6)
        self.place_at_grid(rise_text, 'D4', scale_factor=0.6)
        
        self.play(Create(line_run), Create(line_rise))
        self.play(Write(run_text), Write(rise_text))
        self.play(Indicate(line_rise), Indicate(line_run)) # Lesson L004
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A straight line has the same rate everywhere."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_CHAR)
        )
        
        # Group character, triangle lines, and labels to move as one unit
        climbing_unit = VGroup(character, line_run, line_rise, run_text, rise_text)
        
        # Define destination point further up the ramp
        start_pt = ramp_line.point_from_proportion(0.2)
        dest_pt = ramp_line.point_from_proportion(0.8)
        shift_vec = dest_pt - start_pt
        
        self.play(
            climbing_unit.animate.shift(shift_vec),
            run_time=4,
            rate_func=linear
        )
        
        self.wait(1)
        
        # Final state: reset lecture highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
