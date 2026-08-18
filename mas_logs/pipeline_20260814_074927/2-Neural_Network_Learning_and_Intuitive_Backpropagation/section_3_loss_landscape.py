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

class Section3LossLandscapeScene(TeachingScene):
    def construct(self):
        title = "The Loss Landscape: Walking in the Fog"
        lecture_lines = [
            "Imagine the total error as a hilly landscape.",
            "The height at any point is the loss.",
            "Learning is finding the lowest valley.",
            "We feel the slope to find the way down.",
            "This slope is called the gradient."
        ]
        self.setup_layout(title, lecture_lines)

        # Helper for the loss function
        def loss_func(x):
            # A wavy line that fits within the grid area
            return 0.8 * np.sin(x) + 0.5 * np.cos(2 * x) + 0.2 * x - 1

        # Landscape area roughly B1 to E6
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "stroke_opacity": 0}
        )
        self.place_in_area(axes, "B1", "E6")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        curve = axes.plot(loss_func, x_range=[0, 6], color="#FFFFFF")
        
        # Fog: multiple rectangles with low opacity
        fog_rects = VGroup(*[
            Rectangle(
                width=6, height=5, 
                fill_color="#808080", fill_opacity=0.1, 
                stroke_width=0
            ).move_to(axes.get_center() + np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 0]))
            for _ in range(5)
        ])
        
        self.play(Create(curve), FadeIn(fog_rects))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        x_tracker = ValueTracker(1.0)
        
        # Height arrow using Line + Updaters to avoid expensive mobjects in always_redraw
        height_arrow = Line(color="#FF0000", stroke_width=4)
        def update_arrow(m):
            curr_x = x_tracker.get_value()
            start_p = axes.c2p(curr_x, -3) 
            end_p = axes.c2p(curr_x, loss_func(curr_x))
            m.set_points_as_corners([start_p, end_p])
        
        height_arrow.add_updater(update_arrow)
        
        # Height tip
        height_tip = Triangle(color="#FF0000", fill_opacity=1).scale(0.1)
        def update_tip(m):
            curr_x = x_tracker.get_value()
            m.move_to(axes.c2p(curr_x, loss_func(curr_x)), aligned_edge=DOWN)
        height_tip.add_updater(update_tip)

        height_label = Text("Loss Height", color="#FF0000", font_size=18)
        def update_label(m):
            curr_x = x_tracker.get_value()
            m.next_to(axes.c2p(curr_x, loss_func(curr_x)/2 - 1.5), RIGHT, buff=0.1)
        
        height_label.add_updater(update_label)

        self.play(Create(height_arrow), FadeIn(height_tip), FadeIn(height_label))
        self.play(x_tracker.animate.set_value(4.5), run_time=3, rate_func=linear)
        self.wait(1)
        
        height_arrow.remove_updater(update_arrow)
        height_tip.remove_updater(update_tip)
        height_label.remove_updater(update_label)
        self.play(FadeOut(height_arrow), FadeOut(height_tip), FadeOut(height_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pixel character represented with primitives
        pixel = VGroup(
            Square(side_length=0.3, fill_color="#00FF00", fill_opacity=1, stroke_color=WHITE),
            Dot(radius=0.03, color=BLACK).shift(LEFT*0.06 + UP*0.05),
            Dot(radius=0.03, color=BLACK).shift(RIGHT*0.06 + UP*0.05)
        )
        
        # Initial position on a steep part
        pixel_x = 0.8
        pixel_pos = axes.c2p(pixel_x, loss_func(pixel_x))
        pixel.move_to(pixel_pos)
        
        # Calculate tangent angle
        def get_tangent_angle(x):
            dx = 0.001
            dy = loss_func(x + dx) - loss_func(x)
            return np.arctan2(dy, dx)

        pixel.rotate(get_tangent_angle(pixel_x))

        # Star at global minimum
        min_x = 4.14 
        star = Star(n=5, outer_radius=0.2, inner_radius=0.1, color="#00FF00", fill_opacity=1)
        star.move_to(axes.c2p(min_x, loss_func(min_x)))

        self.play(FadeIn(pixel))
        self.play(Create(star))
        self.play(fog_rects.animate.set_style(fill_opacity=0.03)) 
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Gradient arrow pointing directly UP the slope
        slope = (loss_func(pixel_x + 0.001) - loss_func(pixel_x)) / 0.001
        angle = get_tangent_angle(pixel_x)
        
        grad_arrow = Arrow(ORIGIN, RIGHT * 0.8, color="#FFFF00", buff=0)
        if slope > 0:
            grad_arrow.set_angle(angle)
        else:
            grad_arrow.set_angle(angle + PI)
            
        grad_arrow.move_to(pixel.get_bottom(), aligned_edge=DOWN)
        
        self.play(GrowArrow(grad_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        grad_text = Text("Gradient", color="#FFFF00", font_size=18)
        grad_text.next_to(grad_arrow, UP, buff=0.1)
        
        self.play(Write(grad_text))
        self.wait(2)
