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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        title_text = "Exponential Growth vs. Circular Motion"
        lecture_lines = [
            "Normally, e to the x represents continuous linear growth.",
            "But adding i creates a constant sideways push.",
            "This perpendicular force turns straight growth into a curve.",
            "Continuous sideways motion forms a perfect circular path.",
            "Linear expansion transforms into elegant, eternal rotation."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Visual anchors using the 6x6 grid
        # Center of the circle shifted up as per issue 33
        origin_center = self.grid['C3']
        start_point = self.grid['C4']
        
        # Colors from the animation plan
        BLUE_DOT_COLOR = "#00BFFF"
        MAGENTA_VEC_COLOR = "#FF00FF"
        WHITE_ARC_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_DOT_COLOR))
        
        dot = Dot(color=BLUE_DOT_COLOR)
        # Issue 33: Position dot at C4 (shifted up from D4)
        self.place_at_grid(dot, 'C4', scale_factor=1.0)
        
        growth_label = Text("e^x", color=BLUE_DOT_COLOR, font_size=24)
        # Position label in cell above starting point
        self.place_at_grid(growth_label, 'B4', scale_factor=0.8)
        
        self.play(FadeIn(dot), FadeIn(growth_label))
        
        # Standard horizontal growth demonstration
        self.play(
            dot.animate.move_to(self.grid['C6']),
            growth_label.animate.move_to(self.grid['B6']),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(MAGENTA_VEC_COLOR))
        
        # Return dot to start for the rotation transition
        self.play(
            dot.animate.move_to(start_point),
            growth_label.animate.move_to(self.grid['B4']),
            run_time=1
        )
        
        # Side-push arrow representing 'i'
        i_vec = Arrow(start=start_point, end=self.grid['B4'], color=MAGENTA_VEC_COLOR, buff=0)
        
        # Issue 35: 'i' label (force_label) at B4
        force_label = Text("i", color=MAGENTA_VEC_COLOR, font_size=32)
        self.place_at_grid(force_label, 'B4', scale_factor=0.8)
        force_label.next_to(i_vec.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(i_vec), Write(force_label))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE_ARC_COLOR))
        
        # Visual trace for the resulting circular path
        arc_path = TracedPath(dot.get_center, stroke_color=WHITE_ARC_COLOR, stroke_width=4)
        self.add(arc_path)
        
        angle_tracker = ValueTracker(0)
        
        # Dot updater: follows circle centered at C3 (radius 1 unit in the grid)
        dot.add_updater(lambda d: d.move_to(
            origin_center + np.array([np.cos(angle_tracker.get_value()), np.sin(angle_tracker.get_value()), 0])
        ))
        
        # Vector updater: maintains perpendicularity to path
        def update_i_vec(v):
            pos = dot.get_center()
            angle = angle_tracker.get_value()
            tangent_dir = np.array([-np.sin(angle), np.cos(angle), 0])
            v.become(Arrow(start=pos, end=pos + tangent_dir, color=MAGENTA_VEC_COLOR, buff=0))
        
        i_vec.add_updater(update_i_vec)
        # Force label follows the tip of the vector
        force_label.add_updater(lambda l: l.next_to(i_vec.get_end(), UR, buff=0.1))
        
        self.play(angle_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE_DOT_COLOR))
        
        # Issue 34: e^ix label (growth_formula) positioned in area A4-B5 to avoid collisions
        growth_formula = Text("e^ix", color=BLUE_DOT_COLOR, font_size=28)
        self.place_in_area(growth_formula, 'A4', 'B5', scale_factor=0.7)
        
        self.play(FadeOut(growth_label), FadeIn(growth_formula))
        
        # Continue half-rotation
        self.play(angle_tracker.animate.set_value(PI), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE_ARC_COLOR))
        
        # Complete full circular path
        self.play(angle_tracker.animate.set_value(2*PI), run_time=4, rate_func=linear)
        
        # Stop all updaters
        dot.clear_updaters()
        i_vec.clear_updaters()
        force_label.clear_updaters()
        
        self.wait(2)
