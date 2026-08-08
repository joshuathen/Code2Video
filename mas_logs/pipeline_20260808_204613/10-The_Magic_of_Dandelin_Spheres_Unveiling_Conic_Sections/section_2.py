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
        self.setup_layout("Prerequisite Concept: Tangency and Focus", [
            "A sphere meets a plane at one point.",
            "Tangents from a point to spheres are equal.",
            "This property is our essential key."
        ])
        
        # Elements using assets
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFD700")
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg", color="#32CD32")
        tangent_line = Line(start=LEFT, end=RIGHT, color="#FFD700")
        
        # Grouping for geometric assembly
        geometric_assembly = VGroup(sphere, tangent_line)
        self.place_in_area(geometric_assembly, 'B3', 'D5', scale_factor=0.9)
        
        # Focus/Tangency point
        point_p = Dot(color="#32CD32")
        self.place_at_grid(point_p, 'D4')
        self.place_at_grid(plane, 'F5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sphere), FadeIn(tangent_line))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.play(MoveAlongPath(point_p, tangent_line), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(FadeIn(plane))
        self.play(Indicate(point_p, color="#32CD32"))
