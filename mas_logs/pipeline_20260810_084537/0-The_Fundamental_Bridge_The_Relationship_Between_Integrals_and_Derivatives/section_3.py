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
        lecture_lines = ["The integral is like a painter.", "It accumulates area under the curve.", "A bar sweeps, tracking the total."]
        self.setup_layout("Visualizing the Integral: The Accumulator", lecture_lines)
        
        # Load assets
        paintbrush = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg").scale(0.3)
        
        # Setup Axes
        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 3, 1], 
            axis_config={"include_tip": False}
        ).scale(0.6)
        func = axes.plot(lambda x: 0.2 * (x - 2)**2 + 1, x_range=[0, 4])
        graph_group = VGroup(axes, func)
        # Applying requested adjustments for VideoCritic
        self.place_in_area(graph_group, 'B2', 'D5', scale_factor=0.55)

        # Animation state
        area = axes.get_area(func, x_range=[0, 4], color="#32CD32", opacity=0.5)
        formula = MathTex(r"A = \int_a^b f(x) \, dx", color=WHITE)
        # Applying requested adjustments for VideoCritic
        self.place_at_grid(formula, 'E3', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#32CD32")
        self.play(FadeIn(graph_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        # Adding paintbrush asset interaction
        paintbrush.move_to(graph_group.get_center())
        self.play(FadeIn(area), FadeIn(formula), FadeIn(paintbrush))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        # Visual sweep
        line = Line(axes.c2p(0, 0), axes.c2p(0, 3), color="#32CD32").scale(0.6)
        # Ensure line follows sweep path correctly
        sweep_start = axes.c2p(0, 0)
        sweep_end = axes.c2p(4, 0)
        line.move_to(sweep_start, aligned_edge=DOWN)
        
        self.add(line)
        self.play(
            paintbrush.animate.move_to(axes.c2p(4, 2)), 
            line.animate.shift(RIGHT * (axes.c2p(4, 0)[0] - axes.c2p(0, 0)[0])), 
            run_time=2, 
            rate_func=linear
        )
        self.remove(line, paintbrush)
        self.wait(1)
