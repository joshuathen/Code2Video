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
        self.setup_layout("The Brachistochrone Problem", [
            "Brachistochrone means shortest time in Greek.",
            "What path minimizes travel time between two points?",
            "Straight lines aren't always fastest."
        ])
        
        # Assets
        bead = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg")
        wire = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wire.svg")
        
        point_a = Dot(color=GREEN)
        self.place_at_grid(point_a, "B2", scale_factor=1.0)
        label_a = Text("A", font_size=20, color=GREEN)
        self.place_at_grid(label_a, "A2", scale_factor=0.7)
        
        bead.move_to(point_a.get_center())
        
        point_b = Dot(color=GREEN)
        self.place_at_grid(point_b, "E5", scale_factor=1.0)
        label_b = Text("B", font_size=20, color=GREEN)
        self.place_at_grid(label_b, "E5", scale_factor=0.7)
        
        straight_path = Line(point_a.get_center(), point_b.get_center(), color=YELLOW)
        
        # Bezier curve
        curved_path = CubicBezier(
            point_a.get_center(), 
            point_a.get_center() + DOWN * 1 + RIGHT * 1,
            point_b.get_center() + UP * 1 + LEFT * 1,
            point_b.get_center(), 
            color=PURPLE
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(self.title), Create(point_a), Write(label_a), Create(bead))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(straight_path), Create(point_b), Write(label_b))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        self.play(Create(curved_path))
        self.wait(2)
