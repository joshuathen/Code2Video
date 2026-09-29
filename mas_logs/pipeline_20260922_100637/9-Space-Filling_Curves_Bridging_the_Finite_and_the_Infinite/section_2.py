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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Limit Process", [
            "We start with a simple finite line.", 
            "We repeatedly apply a folding rule.", 
            "The shape grows more complex each step."
        ])
        
        # Load assets
        paper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Define shapes
        step1 = VGroup(Line(LEFT, RIGHT, color=WHITE), paper)
        step1.arrange(DOWN)
        
        step2 = VGroup(*[Line(np.array([i/2-0.5, 0, 0]), np.array([i/2-0.5, 0.5 if i%2==0 else -0.5, 0])) for i in range(3)])
        
        step3 = VGroup(VGroup(*[Line(np.array([i*0.25-1, 0, 0]), np.array([i*0.25-1, 0.3 if i%2==0 else -0.3, 0])) for i in range(8)]), ruler)
        step3.arrange(DOWN)
        
        # Positioning
        self.place_in_area(step1, "B2", "B4", scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(step1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        step2.move_to(self.grid["C3"])
        self.play(Transform(step1, step2))
        self.lecture[1].set_color("#00BFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        step3.move_to(self.grid["D3"])
        self.play(Transform(step1, step3))
        self.lecture[2].set_color("#32CD32")
        self.wait(1)
        
        # Additional visuals
        boundary = Square(side_length=2.5, color="#FF8C00")
        self.place_at_grid(boundary, "D4", scale_factor=1.0)
        self.play(Create(boundary))
        
        # Placeholder for future content
        future_content_placeholder = Circle(radius=0.5, color=YELLOW)
        self.place_in_area(future_content_placeholder, "E3", "F5", scale_factor=0.8)
        self.play(FadeIn(future_content_placeholder))
        
        self.wait(2)
