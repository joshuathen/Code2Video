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
        self.setup_layout("The Familiar Intuition: Real Convergence", [
            "In Euclidean space, smaller steps equal smaller distance.",
            "Example: A squirrel approaching a nut via halves.",
            "The sequence converges as terms approach zero."
        ])
        
        # Assets
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        nut = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nut.svg")
        
        # === Animation for Lecture Line 1 ===
        # Fade in the real number line background #FFFFFF
        number_line = NumberLine(x_range=[0, 1.2, 0.2], length=4, color=WHITE)
        self.place_at_grid(number_line, 'D2', scale_factor=0.9)
        self.place_at_grid(squirrel, 'B3', scale_factor=0.5)
        
        self.play(FadeIn(number_line), FadeIn(squirrel))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Display a point at 1, then 0.5, 0.25 #00FF00
        dots = VGroup(*[Dot(color="#00FF00").move_to(number_line.n2p(1/(2**i))) for i in range(5)])
        self.place_at_grid(nut, 'C4', scale_factor=0.3)
        
        self.play(FadeIn(dots), FadeIn(nut))
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        # Highlight the distance decreasing between points #FF00FF.
        distances = VGroup(*[Line(dots[i].get_center(), dots[i+1].get_center(), color="#FF00FF") for i in range(4)])
        self.play(Create(distances))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
