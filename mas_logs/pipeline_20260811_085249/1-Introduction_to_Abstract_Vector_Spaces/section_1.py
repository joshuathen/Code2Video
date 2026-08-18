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
        lecture_lines = [
            "Vectors aren't just arrows in space.",
            "They follow strict addition rules.",
            "Functions behave just like vectors.",
            "Matrices follow the same addition laws.",
            "Abstraction unlocks new ways to compute."
        ]
        self.setup_layout("From Concrete to Abstract: The Shift", lecture_lines)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        comp = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # Mobjects for animations
        concrete_text = Text("Concrete", font_size=36, color=WHITE)
        abstract_text = Text("Abstract", font_size=36, color=WHITE)
        arrow = Arrow(start=UP, end=DOWN, color=WHITE)
        rectangle_box = SurroundingRectangle(VGroup(concrete_text, abstract_text), color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(ruler, "B5", scale_factor=0.5)
        self.play(FadeIn(ruler))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_at_grid(calc, "C5", scale_factor=0.5)
        self.play(FadeIn(calc))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#87CEEB")
        self.place_at_grid(concrete_text, "B5", scale_factor=0.8)
        self.play(Write(concrete_text))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        self.place_at_grid(arrow, "C5", scale_factor=0.8)
        self.place_at_grid(abstract_text, "D5", scale_factor=0.8)
        self.play(GrowArrow(arrow), FadeIn(abstract_text))
        
        # Add box after structure
        self.place_in_area(rectangle_box, "B3", "D5", scale_factor=0.7)
        self.play(Create(rectangle_box))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#ADFF2F")
        self.place_at_grid(comp, "F5", scale_factor=0.5)
        self.play(FadeIn(comp))
