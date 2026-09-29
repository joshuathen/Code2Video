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
        self.setup_layout("Integral Calculus: The Art of Summation", [
            "Integration is the process of summation.", 
            "We sum infinite parts to find totals.", 
            "Think of filling shapes with thin strips."
        ])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        paper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg")
        
        # Objects
        curve = FunctionGraph(lambda x: 0.1 * x**2 + 0.5, x_range=[-2, 2], color=WHITE)
        self.place_in_area(curve, "A2", "C5")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        rect = Rectangle(width=0.5, height=1.0, color="#FFFFFF", fill_opacity=0.3)
        self.place_at_grid(rect, "D3")
        self.place_at_grid(ruler, "D4", scale_factor=0.5)
        self.play(Create(rect), FadeIn(ruler))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFCC00")
        slices = VGroup(*[Rectangle(width=0.2, height=0.5 + 0.1*i, color="#FFCC00", fill_opacity=0.5) for i in range(5)])
        slices.arrange(RIGHT, buff=0.1).next_to(rect, RIGHT, buff=0.1)
        self.play(Create(slices))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.place_at_grid(paper, "E4", scale_factor=0.8)
        self.play(FadeIn(paper), FadeOut(rect), FadeOut(slices), FadeOut(ruler))
