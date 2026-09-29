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
        self.setup_layout("Summary and Conceptual Leap", [
            "Higher dimensions are found through invariants.", 
            "Look beyond partial 2D truth.", 
            "Expand your perspective to visualize dimensions."
        ])
        
        # Elements
        bird = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        bird.set_color("#9B59B6")
        
        floor = Line(start=np.array([-2, 0, 0]), end=np.array([2, 0, 0]), color=WHITE)
        shadow = Ellipse(width=1, height=0.1, color=GRAY, fill_opacity=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(bird, 'C4', scale_factor=0.7)
        self.play(DrawBorderThenFill(bird))
        self.lecture[0].set_color("#9B59B6")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(floor, 'E3', scale_factor=1.2)
        self.play(Create(floor))
        
        # Shadow projection
        shadow.next_to(floor, DOWN, buff=0.1)
        self.play(FadeIn(shadow))
        self.lecture[1].set_color("#BDC3C7")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        # Bird looks down
        self.play(bird.animate.rotate(0.2, about_point=bird.get_center()))
        self.lecture[2].set_color("#F1C40F")
        self.wait(2)
