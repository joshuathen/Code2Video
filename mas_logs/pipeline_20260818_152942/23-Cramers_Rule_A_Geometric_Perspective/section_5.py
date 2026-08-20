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
        self.setup_layout("Summary & Intuition Check", [
            "Cramer's Rule turns algebra into geometry.", 
            "Problem solving becomes comparing areas.", 
            "It fails when vectors are collinear."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.3)
        self.place_at_grid(axes, 'B3', scale_factor=0.8)
        
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        self.place_at_grid(protractor, 'E1', scale_factor=0.5)
        
        poly = Polygon([0,0,0], [1,1.5,0], [3,2,0], [2,0.5,0], color=YELLOW, fill_opacity=0.3).scale(0.3)
        poly.move_to(axes.c2p(1.5, 1.0))
        
        self.play(Create(axes), Create(protractor), Create(poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        
        area_label = Text("Area Comparison", font_size=20, color=WHITE)
        self.place_at_grid(area_label, 'A3', scale_factor=0.9)
        
        self.play(FadeIn(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=GREEN)
        self.place_at_grid(ruler, 'E6', scale_factor=0.5)
        
        fail_label = Text("Determinant = 0", font_size=20, color=RED)
        self.place_at_grid(fail_label, 'D4', scale_factor=0.9)
        
        self.play(
            poly.animate.scale(0.1),
            FadeIn(ruler),
            FadeIn(fail_label),
            run_time=2
        )
        self.wait(2)
