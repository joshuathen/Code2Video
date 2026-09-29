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
        self.setup_layout("The 2D Pseudo-Cross Product", [
            "2D cross product isn't standard.", 
            "Signed area tracks orientation.", 
            "Positive: counter-clockwise turn."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3388FF"))
        v = Vector([1.5, 0.5], color="#3388FF")
        self.place_at_grid(v, 'C2', scale_factor=0.8)
        v_label = MathTex(r"\vec{v}", color="#3388FF").next_to(v.get_end(), UP, buff=0.1)
        self.place_at_grid(compass, 'A4', scale_factor=0.5)
        self.play(Create(v), Write(v_label), FadeIn(compass))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8833"))
        # Define pseudo-cross product v_perp (rotate 90 counter clockwise)
        v_perp = Vector([-0.5, 1.5], color="#FF8833")
        self.place_at_grid(v_perp, 'C4', scale_factor=0.8)
        v_perp_label = MathTex(r"\vec{v}^\perp", color="#FF8833").next_to(v_perp.get_end(), UP, buff=0.1)
        self.place_at_grid(protractor, 'A5', scale_factor=0.5)
        self.play(Create(v_perp), Write(v_perp_label), FadeIn(protractor))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#88FF33"))
        # Indicate rotation from v to v_perp
        arc = ArcBetweenPoints(v.get_end(), v_perp.get_end(), angle=PI/2, color="#88FF33")
        self.place_in_area(arc, 'B2', 'D4', scale_factor=0.9)
        self.play(Create(arc))
        self.wait(2)
