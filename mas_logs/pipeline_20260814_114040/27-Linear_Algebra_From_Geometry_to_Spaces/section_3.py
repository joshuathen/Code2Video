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
        self.setup_layout("Column Space: Where Can We Go?", [
            "Column space is the reachable territory.",
            "Independent vectors cover the full space.",
            "Dependent vectors collapse space into lower dimensions."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        
        origin = self.grid["D3"]
        v1 = Arrow(origin, origin + np.array([1.5, 1, 0]), color="#FF5733", buff=0)
        v2 = Arrow(origin, origin + np.array([0.5, -1.5, 0]), color="#33FF57", buff=0)
        v3 = Arrow(origin, origin + np.array([1, -0.2, 0]), color="#FFFF33", buff=0)
        
        vectors_group = VGroup(v1, v2, compass)
        
        span = Polygon(
            origin, 
            origin + np.array([1.5, 1, 0]), 
            origin + np.array([2.0, -0.5, 0]), 
            origin + np.array([0.5, -1.5, 0]), 
            color="#33A1FF", fill_opacity=0.3
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.place_in_area(vectors_group, 'A4', 'F6', scale_factor=0.6)
        self.place_at_grid(compass, 'A6', scale_factor=0.5)
        self.play(Create(v1), Create(v2), FadeIn(compass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.play(FadeIn(span))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33A1FF")
        self.place_at_grid(map_icon, 'F6', scale_factor=0.5)
        self.play(Create(v3), FadeIn(map_icon))
        self.wait(2)
