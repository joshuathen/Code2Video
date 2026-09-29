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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Gaussian Bell Curve", ["Look at the standard normal curve.", "Its function contains π.", "Why is π in linear distributions?"])
        
        # Define objects
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), color=WHITE)
        formula = MathTex(r"f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}").scale(0.7)
        question_mark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Positioning
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.45)
        bell_curve.replace(axes.plot(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), color=WHITE)) # Reset scale
        self.place_in_area(bell_curve, 'B3', 'F6', scale_factor=0.45)
        
        self.place_at_grid(formula, 'B2', scale_factor=0.6)
        self.place_at_grid(question_mark, 'C4', scale_factor=0.3)
        question_mark.next_to(formula, RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), FadeIn(bell_curve))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(Write(formula))
        self.lecture[1].set_color(PINK)
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(question_mark))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
