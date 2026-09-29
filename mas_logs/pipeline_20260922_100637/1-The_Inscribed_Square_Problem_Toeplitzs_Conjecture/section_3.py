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
        lecture_lines = ["We use the Borsuk-Ulam Theorem.", "It relates spheres and flat planes.", "Opposite points must have matching values.", "This forces a square to emerge.", "The topology guarantees a solution exists."]
        self.setup_layout("The Mathematical Mechanism: Symmetry & Topology", lecture_lines)
        
        # Load Assets
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        nodes = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nodes.svg")
        
        # Prepare for area placement
        circle_square_group = VGroup(sphere, square)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), run_time=0.5)
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(sphere, 'B4', scale_factor=0.6)
        self.play(FadeIn(self.lecture[1]), FadeIn(sphere), Rotate(sphere, angle=2*PI), run_time=1)
        self.lecture[1].set_color("#ADD8E6")
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]), run_time=1)
        self.lecture[2].set_color("#FFD700")
        
        # === Animation for Lecture Line 4 ===
        self.place_at_grid(square, 'E4', scale_factor=0.4)
        self.play(FadeIn(self.lecture[3]), FadeIn(square), Transform(square, triangle), run_time=1)
        self.lecture[3].set_color("#FF4500")
        
        # === Animation for Lecture Line 5 ===
        self.place_in_area(VGroup(sphere, square), 'B4', 'E4', scale_factor=0.7)
        self.place_at_grid(nodes, 'C4', scale_factor=0.5)
        nodes.set_color("#00CED1")
        self.play(FadeIn(self.lecture[4]), FadeIn(nodes), run_time=1)
        self.lecture[4].set_color("#00CED1")
        
        self.wait(2)
