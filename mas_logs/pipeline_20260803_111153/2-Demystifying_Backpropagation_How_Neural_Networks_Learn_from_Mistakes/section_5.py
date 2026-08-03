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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Imagine a landscape where height represents the cost.",
            "The gradient points in the steepest uphill direction.",
            "We step in the opposite direction to go downhill.",
            "Like a ball rolling toward the lowest valley point.",
            "Reaching the bottom minimizes our total prediction error."
        ]
        self.setup_layout("Gradient Descent: The Downhill Walk", lecture_lines)

        # Parabola parameters: y = a(x-h)^2 + k
        # Centered at x=3.0 (Grid cols 3-4), bottom at y=-1.8 (Grid row E)
        h, k, a = 3.0, -1.8, 0.6
        
        def parabola_func(x):
            return a * (x - h)**2 + k

        def get_tangent_vector(x, length=0.8):
            # Numeric derivative for tangent calculation
            dx = 0.001
            slope = (parabola_func(x + dx) - parabola_func(x)) / dx
            direction = np.array([1, slope, 0])
            norm = np.linalg.norm(direction)
            if norm == 0: return np.array([length, 0, 0])
            return (direction / norm) * length

        # === Animation for Lecture Line 1 ===
        # Imagine a landscape where height represents the cost.
        curve = FunctionGraph(
            parabola_func,
            x_range=[1.2, 4.8],
            color="#00FF00"
        )
        
        # Initial ball position at x=1.5
        x_tracker = ValueTracker(1.5)
        ball = Dot(color="#FFFFFF", radius=0.15)
        # Persistent updater for the ball
        ball.add_updater(lambda m: m.move_to([x_tracker.get_value(), parabola_func(x_tracker.get_value()), 0]))
        
        self.lecture[0].set_color("#00FF00")
        self.play(Create(curve), FadeIn(ball))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The gradient points in the steepest uphill direction.
        uphill_arrow = Arrow(color="#FFFF00", buff=0, stroke_width=5)
        def update_uphill(m):
            x = x_tracker.get_value()
            start = ball.get_center()
            vec = get_tangent_vector(x)
            # Uphill is opposite to downhill. If x < h, downhill is right, so uphill is left.
            if x < h:
                vec = -vec
            m.put_start_and_end_on(start, start + vec)
        
        uphill_arrow.add_updater(update_uphill)
        
        self.lecture[1].set_color("#FFFF00")
        self.play(GrowArrow(uphill_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We step in the opposite direction to go downhill.
        # Cyan dashed arrow pointing down the slope.
        downhill_arrow = DashedLine(color="#00FFFF", dash_length=0.1).add_tip(tip_length=0.2)
        def update_downhill(m):
            x = x_tracker.get_value()
            start = ball.get_center()
            vec = get_tangent_vector(x)
            # Downhill: if x > h, downhill is left.
            if x > h:
                vec = -vec
            # For DashedLine with tip, we update point by point
            m.set_points_by_ends(start, start + vec)
            
        downhill_arrow.add_updater(update_downhill)
        
        self.lecture[2].set_color("#00FFFF")
        self.play(Create(downhill_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Like a ball rolling toward the lowest valley point.
        self.lecture[3].set_color("#FFFFFF")
        # Ball rolls as x_tracker moves toward the minimum at h=3.0
        self.play(
            x_tracker.animate.set_value(h),
            run_time=3,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Reaching the bottom minimizes our total prediction error.
        # Bottom of the curve glows gold.
        glow = Dot(point=[h, k, 0], color="#FFD700", radius=0.4).set_opacity(0)
        
        self.lecture[4].set_color("#FFD700")
        self.play(
            glow.animate.set_opacity(0.6).scale(1.5),
            curve.animate.set_color("#FFD700"),
            FadeOut(uphill_arrow),
            FadeOut(downhill_arrow)
        )
        # Success indicator flash
        flash_target = Dot(point=[h, k, 0], color="#FFD700").set_opacity(0)
        self.add(flash_target)
        self.play(Flash(flash_target, color="#FFD700", line_length=0.3))
        self.wait(2)
