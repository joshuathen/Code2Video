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
        self.setup_layout("Defining the Cross Product: Direction and Magnitude", 
                          ["The cross product creates a new 3D vector.", 
                           "Its length equals the parallelogram's area.", 
                           "Direction follows the right-hand rule."])
        
        # Setup 3D elements
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)
        
        # Create vectors
        vec_a = Arrow(ORIGIN, [1, 0.5, 0], buff=0, color=BLUE)
        vec_b = Arrow(ORIGIN, [0, 1, 0.5], buff=0, color=GREEN)
        vec_p = Arrow(ORIGIN, [0, 0, 1.5], buff=0, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(vec_a), Create(vec_b))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Parallelogram visual
        para = Polygon(ORIGIN, [1, 0.5, 0], [1, 1.5, 0.5], [0, 1, 0.5], color=WHITE, fill_opacity=0.3)
        self.play(FadeIn(para))
        self.play(Create(vec_p))
        label_p = MathTex(r"\\vec{A} \\times \\vec{B}", color=YELLOW).scale(0.8)
        self.place_at_grid(label_p, 'C2', scale_factor=0.7)
        self.play(Write(label_p))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF33"))
        
        # Right hand rule gesture - using asset
        gesture = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        self.place_at_grid(gesture, 'D5', scale_factor=0.7)
        self.play(FadeIn(gesture))
        self.play(gesture.animate.shift(UP * 0.2), run_time=0.5)
        self.play(gesture.animate.shift(DOWN * 0.2), run_time=0.5)
        self.wait(2)
