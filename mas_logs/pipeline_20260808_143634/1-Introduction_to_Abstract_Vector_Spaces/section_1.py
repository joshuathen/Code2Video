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
            "Vectors extend beyond simple geometric arrows.",
            "Objects are vectors if they follow specific rules.",
            "We study addition and scaling behavior.",
            "Abstract spaces mirror intuitive vector movement.",
            "A set becomes a space via these rules."
        ]
        self.setup_layout("From Concrete to Abstract: The Intuition", lecture_lines)
        
        # Define Mobjects
        vector = Arrow(ORIGIN, UP*1 + RIGHT*1, color=WHITE)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        vector_group = VGroup(vector, ruler)
        
        abstract_dot = Dot(color="#00FF00")
        abstract_space = VGroup(*[Dot(color="#FF00FF").move_to(self.grid[pos]) for pos in ["B2", "B4", "C3", "D2", "D4"]])
        arrows = VGroup(
            Arrow(abstract_space[0].get_center(), abstract_space[1].get_center(), color="#FFFF00", buff=0.1),
            Arrow(abstract_space[2].get_center(), abstract_space[3].get_center(), color="#FFFF00", buff=0.1)
        )
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(vector_group, 'B2', 'D2', scale_factor=0.7)
        self.play(Create(vector_group))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(abstract_dot, 'D5', scale_factor=1.5)
        label = Text('Abstract', font_size=24).move_to(self.grid['D6'])
        self.add(label)
        self.play(ReplacementTransform(vector_group, abstract_dot))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(abstract_space))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(Create(arrows))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(compass, 'A1', scale_factor=0.8)
        self.play(
            FadeOut(abstract_space), 
            FadeOut(arrows), 
            FadeOut(abstract_dot),
            FadeOut(label),
            FadeIn(compass)
        )
        self.lecture[4].set_color(WHITE)
        self.wait(1)
