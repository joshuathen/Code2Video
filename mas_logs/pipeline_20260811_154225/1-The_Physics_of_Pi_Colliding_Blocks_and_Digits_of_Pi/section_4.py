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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The total path angle is Pi radians.", "Collisions count how many arcs fit.", "Pi emerges from the geometry of constraints."]
        self.setup_layout("The Geometric Proof: Why Pi?", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        self.place_in_area(circle, 'B4', 'D6', scale_factor=0.6)
        self.play(Create(circle))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347")
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/arc.svg
        arc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arc.svg", color="#FF6347")
        self.place_in_area(arc, 'B4', 'D6', scale_factor=0.6)
        
        # Gauge hand - manually keeping it as a Line but using proper grid
        gauge = Line(ORIGIN, UP*1.2, color="#FFD700")
        self.place_in_area(gauge, 'B4', 'D6', scale_factor=0.6)
        self.play(Create(arc), Create(gauge))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/geometry.svg
        geom = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/geometry.svg", color="#FFFF00")
        self.place_in_area(geom, 'E4', 'F6', scale_factor=0.5)
        
        pi_val = MathTex(r"\\pi \\approx 3.1415", color="#FFD700")
        self.place_at_grid(pi_val, 'F4', scale_factor=0.9)
        self.play(FadeIn(geom), Write(pi_val))
        self.wait(2)
