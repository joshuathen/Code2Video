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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines_text = [
            "Evaluate integral via antiderivative subtraction.",
            "F(b) minus F(a) equals area.",
            "Simplifies complex area into subtraction.",
            "Robotic turtle position example logic.",
            "Practical computation made easy."
        ]
        self.setup_layout("The Fundamental Theorem of Calculus (Part 2)", lecture_lines_text)
        
        # Math objects
        formula = MathTex(r"\int_{a}^{b} f(x) \, dx = F(b) - F(a)", font_size=36)
        
        # Axes
        axes = Axes(
            x_range=[0, 6, 1], y_range=[0, 4, 1],
            axis_config={"include_numbers": False}
        )
        graph = axes.plot(lambda x: 0.2*(x-3)**2 + 1, x_range=[0.5, 5.5], color=YELLOW)
        
        # Area under curve
        area = axes.get_area(graph, x_range=[1.5, 4.5], color=BLUE, opacity=0.3)
        
        # Labels
        label_a = MathTex("a", color="#FF0000").next_to(axes.c2p(1.5, 0), DOWN)
        label_b = MathTex("b", color="#00FF00").next_to(axes.c2p(4.5, 0), DOWN)
        
        # Turtle asset
        turtle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turtle.svg", color="#FFA500")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(formula, "B4", scale_factor=1.0)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_in_area(axes, "D2", "E5", scale_factor=0.5)
        self.play(Create(axes), Create(graph))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(Create(area), Write(label_a), Write(label_b))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFA500"))
        # Using SVGMobject from Asset, applying requested grid placement
        self.place_at_grid(turtle, "F2", scale_factor=0.7)
        self.play(FadeIn(turtle))
        # Move turtle across the bottom as an indicator
        self.play(turtle.animate.move_to(self.grid["F5"]), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(FadeOut(turtle))
