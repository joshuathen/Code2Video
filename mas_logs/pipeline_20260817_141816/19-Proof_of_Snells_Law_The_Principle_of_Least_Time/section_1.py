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
        lecture_lines = ["Light travels the path of least time.", "A lifeguard on sand runs faster than swimming.", "They choose an angled path to save time."]
        self.setup_layout("Fermat's Principle", lecture_lines)
        
        # Define visual elements
        # Assets
        lifeguard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lifeguard.svg")
        sand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg")
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        
        A = Dot(color=WHITE)
        B = Dot(color=WHITE)
        
        # Place assets and dots
        self.place_at_grid(lifeguard, 'B2', scale_factor=0.3)
        self.place_at_grid(sand, 'D3', scale_factor=0.3)
        self.place_at_grid(water, 'E5', scale_factor=0.3)
        
        self.place_at_grid(A, 'B2', scale_factor=0.8)
        self.place_at_grid(B, 'E5', scale_factor=0.8)
        
        # Straight path
        straight_path = Line(A.get_center(), B.get_center(), color=WHITE)
        
        # Curved/Angled path
        curve_pts = [A.get_center(), self.grid['C3'], B.get_center()]
        angled_path = VMobject(color=WHITE)
        angled_path.set_points_smoothly(curve_pts)
        
        path_group = VGroup(straight_path, angled_path)
        self.place_in_area(path_group, 'B2', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.title))
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(lifeguard), Create(A), Create(B))
        self.play(Create(straight_path.set_color("#FFD700")))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(FadeIn(sand))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.play(FadeIn(water))
        self.play(ReplacementTransform(straight_path, angled_path.set_color("#FF4500")))
        self.wait(2)
