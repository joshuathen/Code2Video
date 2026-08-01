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
        # Setup Layout
        lines = [
            'The length of v cross w represents an area.',
            'It equals the area of the parallelogram they form.',
            'As the angle changes, the area follows sine theta.',
            'If vectors are parallel, the area and product vanish.',
            'This magnitude scales with the span of the vectors.'
        ]
        self.setup_layout("The Magnitude: Measuring the Parallelogram", lines)

        # Vector parameters
        v_color = "#58C4DD"
        w_color = "#83C167"
        n_color = "#F8B195"
        area_color = "#FFFFFF"
        label_color = "#FFFF00"

        # Grid-based origin for vectors
        origin = self.grid["D2"]
        
        # Trackers
        theta_tracker = ValueTracker(30 * DEGREES)
        v_length = 1.5
        w_length = 1.2

        # Vectors
        v_vec = Arrow(origin, origin + RIGHT * v_length, buff=0, color=v_color)
        w_vec = Arrow(origin, origin + UP * w_length, buff=0, color=w_color) # Initial w
        
        def get_w_end():
            angle = theta_tracker.get_value()
            return origin + np.array([np.cos(angle) * w_length, np.sin(angle) * w_length, 0])

        w_vec.add_updater(lambda m: m.become(Arrow(origin, get_w_end(), buff=0, color=w_color)))

        # Parallelogram
        parallelogram = Polygon(
            origin,
            origin + RIGHT * v_length,
            origin + RIGHT * v_length + (get_w_end() - origin),
            get_w_end(),
            fill_opacity=0.2,
            fill_color=area_color,
            stroke_width=1,
            color=area_color
        )
        
        def update_para(p):
            w_end = get_w_end()
            p.set_points_as_corners([
                origin,
                origin + RIGHT * v_length,
                origin + RIGHT * v_length + (w_end - origin),
                w_end,
                origin
            ])

        parallelogram.add_updater(update_para)

        # Cross product vector n (vertical representation for visualization)
        n_origin = self.grid["C5"]
        n_vec = Arrow(n_origin, n_origin + UP * 0.1, buff=0, color=n_color)
        
        def update_n(n):
            angle = theta_tracker.get_value()
            area_val = v_length * w_length * np.abs(np.sin(angle))
            # Scale height of n relative to area
            n.become(Arrow(n_origin, n_origin + UP * area_val, buff=0, color=n_color))
        
        n_vec.add_updater(update_n)

        # Labels
        area_label = Text("Area = |v||w| sin(θ)", font_size=18, color=label_color)
        self.place_at_grid(area_label, "B3")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(v_color)
        self.play(GrowArrow(v_vec), GrowArrow(w_vec), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(area_color)
        self.play(FadeIn(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(label_color)
        self.play(
            theta_tracker.animate.set_value(75 * DEGREES),
            FadeIn(area_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        self.play(
            theta_tracker.animate.set_value(0.001 * DEGREES),
            area_label.animate.set_opacity(0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(n_color)
        n_text = Text("|v × w|", font_size=18, color=n_color)
        self.place_at_grid(n_text, "D5")
        
        self.play(
            theta_tracker.animate.set_value(90 * DEGREES),
            area_label.animate.set_opacity(1),
            FadeIn(n_vec),
            Write(n_text),
            run_time=2
        )
        self.wait(3)
