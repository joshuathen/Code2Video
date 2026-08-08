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
        self.setup_layout("Introduction: The Fastest Path", [
            "Brachistochrone means shortest time in Greek.",
            "Bead slides from A to B under gravity.",
            "Which path shape minimizes travel time?"
        ])
        
        # Assets
        bead_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg"
        
        # Grid Labels
        grid_labels = Text("Grid Layout", font_size=20, color=BLUE)
        self.place_at_grid(grid_labels, 'D4', scale_factor=0.7)
        
        # Header
        header_text = Text("Brachistochrone Problem", font_size=24, color=YELLOW)
        self.place_at_grid(header_text, 'A3', scale_factor=0.9)
        
        # Define Points A and B for animation path
        A = self.grid['B2']
        B = self.grid['E5']

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        
        # Load asset
        bead = SVGMobject(bead_asset, color=YELLOW).scale(0.05)
        
        # Create container for path to be placed in area
        path_container = VGroup()
        line = Line(A, B, color=WHITE)
        dot_a = Dot(A, color=RED)
        dot_b = Dot(B, color=RED)
        path_container.add(line, dot_a, dot_b)
        
        self.place_in_area(path_container, 'A1', 'F3', scale_factor=0.6)
        self.add(path_container)
        
        bead.move_to(A)
        self.play(FadeIn(bead))
        self.play(MoveAlongPath(bead, line), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF69B4"))
        
        # Curve path (Cubic Bezier)
        curve = CubicBezier(A, A + DOWN + RIGHT, B + UP + LEFT, B, color=WHITE)
        self.play(Create(curve))
        
        bead.move_to(A)
        self.play(MoveAlongPath(bead, curve), run_time=1.5)
        self.wait(1)
