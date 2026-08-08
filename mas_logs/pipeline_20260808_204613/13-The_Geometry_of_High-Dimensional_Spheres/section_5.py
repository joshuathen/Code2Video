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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application & Summary", [
            "Data science uses distance for classification.",
            "Hyperspheres cluster similar user preferences efficiently.",
            "High-dimensional spaces defy our 3D intuition."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Prerequisites: Show a scatter plot of high-dimensional points.
        dots = VGroup(*[Dot(color=BLUE) for _ in range(20)])
        dots.arrange_in_grid(4, 5, buff=0.1)
        self.place_in_area(dots, 'D1', 'E3', scale_factor=0.5)
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        self.place_at_grid(computer_icon, 'A2', scale_factor=0.4)
        
        self.play(FadeIn(dots), FadeIn(computer_icon), self.lecture[0].animate.set_color("#FF69B4"))

        # === Animation for Lecture Line 2 ===
        # Formalization: Display the KNN search formula.
        formula = MathTex(r"d(x, y) = \sqrt{\sum (x_i - y_i)^2}", color=WHITE)
        self.place_at_grid(formula, 'B4', scale_factor=0.9)
        self.play(Write(formula), self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        # Real-world: Animate a search engine clustering points.
        circle = Circle(radius=1.0, color=WHITE)
        browser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/browser.svg", color=WHITE)
        self.place_at_grid(circle, 'F5', scale_factor=0.6)
        self.place_at_grid(browser_icon, 'F6', scale_factor=0.4)
        
        self.play(Create(circle), FadeIn(browser_icon), self.lecture[2].animate.set_color("#32CD32"))
        self.wait(2)
