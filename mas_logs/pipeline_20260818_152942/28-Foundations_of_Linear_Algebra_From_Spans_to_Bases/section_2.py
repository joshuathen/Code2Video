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
        self.setup_layout("Linear Combinations and Span", ["Linear combinations combine scaled vectors.", "Span is all reachable points.", "Different directions cover the plane."])
        
        # === Animation for Lecture Line 1 ===
        # 1. Display two basis vectors on the plane [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg]. Color: #FFD700 (Gold).
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.5)
        self.add(plane)
        
        v1 = Arrow(ORIGIN, RIGHT + UP, color="#FFD700")
        v2 = Arrow(ORIGIN, RIGHT * 0.5 + DOWN, color="#FFD700")
        self.place_at_grid(v1, 'C2', scale_factor=0.9)
        self.place_at_grid(v2, 'D2', scale_factor=0.8)
        self.play(Create(v1), Create(v2))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        # 2. Draw linear combination as path of two vectors. Color: #00BFFF (Deep Sky Blue).
        path = VGroup(v1.copy(), v2.copy().shift(v1.get_end()))
        self.play(Create(path), run_time=2)
        self.lecture[1].set_color("#00BFFF")
        
        # 3. Highlight resulting grid points. Color: #FF4500 (Orange Red).
        dot = Dot(path[1].get_end(), color="#FF4500")
        self.play(FadeIn(dot))
        
        # 4. Fill the span area with light shading. Color: #32CD32 (Lime Green).
        span_area = Polygon(
            np.array([-1, -1, 0]), np.array([1, -1, 0]),
            np.array([1, 1, 0]), np.array([-1, 1, 0]),
            color="#32CD32", fill_opacity=0.2
        )
        self.place_in_area(span_area, 'A4', 'F6', scale_factor=0.6)
        self.play(FadeIn(span_area))
        
        # === Animation for Lecture Line 3 ===
        # 5. Show scaling basis vectors to fill plane [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg]. Color: #FF1493 (Deep Pink).
        self.play(v1.animate.scale(1.5), v2.animate.scale(1.5))
        self.lecture[2].set_color("#FF1493")
        self.wait(2)
