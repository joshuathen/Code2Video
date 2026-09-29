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
        self.setup_layout("Philosophical Synthesis: Reconciling Finite and Infinite", [
            "The curve reaches infinite density in the limit.",
            "Mathematics transforms the finite into a new existence.",
            "Fractal antennas pack length into tiny, finite spaces."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        title_box = Text("Infinite Path, Finite Area", font_size=32, color="#FFFFFF")
        # Issue 26 Fix: Grid A4
        self.place_at_grid(title_box, 'A4', scale_factor=0.9)
        self.play(FadeIn(title_box))
        
        # Issue 27 Fix: Area D4-F6
        boundary = Square(side_length=1.5, color="#00BFFF")
        self.place_in_area(boundary, 'D4', 'F6', scale_factor=0.8)
        self.play(Create(boundary))
        
        # === Animation for Lecture Line 2 ===
        # Integrate [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg]
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        antenna = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg", color="#FFFFFF")
        self.place_at_grid(antenna, 'C4', scale_factor=0.6)
        
        path = VMobject(color="#FF4500")
        path.set_points_smoothly([boundary.get_center() + np.array([-0.5, -0.5, 0]), 
                                 boundary.get_center() + np.array([0, 0.5, 0]), 
                                 boundary.get_center() + np.array([0.5, -0.5, 0])])
        self.play(FadeIn(antenna), Create(path))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        conclusion = Text("Dimension is not just space", font_size=24, color="#ADFF2F")
        # Issue 28 Fix: Grid B6
        self.place_at_grid(conclusion, 'B6', scale_factor=0.7)
        self.play(FadeIn(conclusion))
        self.wait(2)
