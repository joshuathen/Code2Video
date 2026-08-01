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
        self.setup_layout("The Error (Loss) Function", [
            "The loss function measures our distance from success.",
            "Visualize this as a valley with a bottom.",
            "The lowest point represents zero error and perfection."
        ])
        
        # Colors
        color_parabola = "#1E90FF"
        color_ball = "#FF4500"
        color_error = "#FF4500"
        color_weight = "#00FF00"
        
        # Assets
        ball_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"

        # === Animation for Lecture Line 1 ===
        # The loss function measures our distance from success.
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(
            x_range=[-2.2, 2.2, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4,
            y_length=4
        )
        
        weight_label = Text("Weight", font_size=18, color=color_weight)
        error_label = Text("Error", font_size=18, color=color_error)
        
        parabola = axes.plot(lambda x: x**2, x_range=[-2, 2], color=color_parabola)
        
        graph_group = VGroup(axes, parabola)
        # Resolved Issue 35 & 36: Adjusted area and scale to prevent overlap with title and edge
        self.place_in_area(graph_group, "B2", "F6", scale_factor=0.8)
        
        # Position labels relative to axes after placement
        error_label.next_to(axes.y_axis, UP, buff=0.2)
        weight_label.next_to(axes.x_axis, RIGHT, buff=0.2)

        self.play(Create(axes), Write(error_label), Write(weight_label))
        self.play(Create(parabola))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualize this as a valley with a bottom.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        ball_start_x = 1.6
        ball_pos = axes.c2p(ball_start_x, ball_start_x**2)
        
        # Resolved Issue 28: Using provided SVG asset for the ball
        ball = SVGMobject(ball_asset)
        ball.set_color(color_ball)
        ball.scale(0.15) # Adjusted for visual balance
        ball.move_to(ball_pos)
        
        # Vertical distance line representing error
        dist_line = Line(
            start=axes.c2p(ball_start_x, 0),
            end=ball_pos,
            color=color_error,
            stroke_width=4
        )
        dist_text = Text("Gap", font_size=16, color=color_error).next_to(dist_line, LEFT, buff=0.1)

        self.play(FadeIn(ball), Create(dist_line), Write(dist_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The lowest point represents zero error and perfection.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Value tracker for rolling down the slope
        t = ValueTracker(ball_start_x)
        
        # Persistent updaters for smooth motion
        ball.add_updater(lambda m: m.move_to(axes.c2p(t.get_value(), t.get_value()**2)))
        dist_line.add_updater(lambda m: m.set_points_as_corners([
            axes.c2p(t.get_value(), 0),
            axes.c2p(t.get_value(), t.get_value()**2)
        ]))
        dist_text.add_updater(lambda m: m.next_to(dist_line, LEFT, buff=0.1))

        # Roll the ball to the bottom
        self.play(
            t.animate.set_value(0),
            run_time=3,
            rate_func=smooth
        )
        
        # Final highlight at the zero-error point
        ball.clear_updaters()
        dist_line.clear_updaters()
        dist_text.clear_updaters()
        
        self.play(
            ball.animate.scale(1.5).set_color(GOLD),
            FadeOut(dist_line),
            FadeOut(dist_text)
        )
        
        self.wait(2)
