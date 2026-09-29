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
        lecture_lines = [
            "Julia sets define boundaries of dynamic behavior.",
            "Small changes lead to vastly different outcomes.",
            "These boundaries form complex, beautiful patterns.",
            "Points flee to infinity or spiral toward attractors.",
            "This sensitive dependence creates chaotic boundaries."
        ]
        self.setup_layout("The Julia Set: Defining the Boundary of Chaos", lecture_lines)
        
        # Asset: Ruler
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Animations
        plane = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False})
        # Applying fix: place plane in C2-F6
        self.place_in_area(plane, "C2", "F6", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(plane), FadeIn(ruler), run_time=1)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        diverging_points = VGroup(*[Dot(plane.c2p(0.5+0.1*i, 0.5+0.1*i), color="#FF0000", radius=0.03) for i in range(5)])
        self.play(FadeIn(diverging_points), run_time=1)
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        bounded_points = VGroup(*[Dot(plane.c2p(-0.2+0.05*i, -0.2+0.05*i), color="#0000FF", radius=0.03) for i in range(5)])
        self.play(FadeIn(bounded_points), run_time=1)
        self.lecture[2].set_color("#0000FF")

        # === Animation for Lecture Line 4 ===
        boundary = Circle(radius=0.5, color="#FFFFFF", stroke_width=2).move_to(plane.c2p(0, 0))
        self.play(Create(boundary), run_time=1)
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        # Asset: Magnifying glass
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        self.place_at_grid(magnifier, 'C4', scale_factor=0.5)
        
        highlight = Circle(radius=0.5, color="#FFFF00", stroke_width=4).move_to(plane.c2p(0, 0))
        self.play(FadeIn(magnifier), Indicate(highlight), run_time=1)
        self.lecture[4].set_color("#FFFF00")
