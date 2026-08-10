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
        self.setup_layout("Linear Dependence: The Redundancy Rule", [
            "Linear dependence means redundancy exists.",
            "Dependent vectors share same line or plane.",
            "Redundant vectors add no new directions."
        ])
        
        # Define vectors
        u = Arrow(ORIGIN, RIGHT * 1.5, color="#FF6666")
        v = Arrow(ORIGIN, UP * 1.5, color="#66FF66")
        w = Arrow(ORIGIN, (RIGHT + UP) * 1.5, color="#6666FF")
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        v_group = VGroup(u, v, w, ruler)
        self.place_in_area(v_group, 'B2', 'D5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(u, v, w, ruler))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate w to align with u+v
        target_w = Arrow(ORIGIN, (RIGHT + UP) * 2.12, color="#6666FF")
        self.play(w.animate.put_start_and_end_on(ORIGIN, (RIGHT + UP) * 2.12))
        self.lecture[1].set_color(YELLOW)
        
        dep_text = Text("Dependent", color="#FF0000", font_size=32)
        self.place_at_grid(dep_text, 'E4')
        self.place_at_grid(protractor, 'E5', scale_factor=0.8)
        self.play(FadeIn(dep_text, protractor))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
