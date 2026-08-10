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
        self.setup_layout("Visualizing Closure: The Gatekeeper", [
            "Closure is our most important filter.", 
            "Operations must keep elements within the space.",
            "Gatekeeper_Fence_Animation"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight "Closure" property tag
        closure_text = Text("CLOSURE", font_size=36, color=YELLOW).add_background_rectangle()
        self.place_at_grid(closure_text, 'C2', scale_factor=0.9)
        self.play(FadeIn(closure_text))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Load asset and use it as visual boundary
        fence = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fence.svg")
        self.place_in_area(fence, 'C4', 'E6', scale_factor=0.85)
        
        # Visualise vector addition
        # Vectors within the fence area
        v1 = Arrow(start=ORIGIN, end=RIGHT*0.8 + UP*0.3, color=ORANGE, buff=0)
        v2 = Arrow(start=RIGHT*0.8 + UP*0.3, end=RIGHT*0.3 + UP*0.8, color=ORANGE, buff=0)
        
        vec_group = VGroup(v1, v2)
        vec_group.move_to(fence.get_center())
        
        self.play(Create(fence), Create(v1), Create(v2))
        self.play(self.lecture[1].animate.set_color(ORANGE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Resultant vector in color #00FF00
        v_sum = Arrow(start=ORIGIN, end=RIGHT*0.3 + UP*0.8, color=GREEN, buff=0)
        v_sum.move_to(fence.get_center())
        
        self.play(Create(v_sum))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
