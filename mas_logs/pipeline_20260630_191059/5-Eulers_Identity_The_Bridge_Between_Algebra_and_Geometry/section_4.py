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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        lecture_lines = [
            "The term e to the i x tracks rotation.",
            "Its position is defined by cosine and sine values.",
            "This formula shows how exponents and circles are related."
        ]
        
        self.setup_layout("The Formula in Motion: e^(ix) as a Rotation", lecture_lines)
        
        # Colors from storyboard
        color_circle = "#808080"  # Grey
        color_point = "#FFFF00"   # Yellow
        color_cos = "#00FF00"      # Green
        color_sin = "#FF0000"      # Red
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_point)
        
        # Grey unit circle (#808080)
        unit_circle = Circle(radius=1.5, color=color_circle)
        self.place_in_area(unit_circle, "B2", "E5")
        center = unit_circle.get_center()
        
        # Faint axes for context
        axes = Axes(
            x_range=[-1.2, 1.2],
            y_range=[-1.2, 1.2],
            x_length=3.0,
            y_length=3.0,
            axis_config={"color": GREY, "stroke_width": 1},
            tips=False
        ).move_to(center)
        
        # Yellow point (#FFFF00)
        point_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/point.svg")
        point_asset.set_color(color_point).scale(0.15)
        
        angle_tracker = ValueTracker(0)
        
        # Update point position based on angle
        point_asset.add_updater(lambda m: m.move_to(
            center + np.array([
                1.5 * np.cos(angle_tracker.get_value()),
                1.5 * np.sin(angle_tracker.get_value()),
                0
            ])
        ))
        
        self.add(axes)
        self.play(Create(unit_circle))
        self.play(FadeIn(point_asset))
        # Start rotation
        self.play(angle_tracker.animate.set_value(PI/4), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_cos)
        
        # Projections
        # Horizontal 'cos(x)' in green on X-axis
        cos_line = Line(center, center, color=color_cos, stroke_width=6)
        # Vertical 'i sin(x)' in red on Y-axis
        sin_line = Line(center, center, color=color_sin, stroke_width=6)
        
        cos_line.add_updater(lambda l: l.put_start_and_end_on(
            center,
            center + np.array([1.5 * np.cos(angle_tracker.get_value()), 0, 0])
        ))
        
        sin_line.add_updater(lambda l: l.put_start_and_end_on(
            center,
            center + np.array([0, 1.5 * np.sin(angle_tracker.get_value()), 0])
        ))
        
        # Connector dashed lines from point to axes
        dashed_h = DashedLine(color=GRAY_C, stroke_width=2)
        dashed_v = DashedLine(color=GRAY_C, stroke_width=2)
