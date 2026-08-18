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
        lecture_lines_text = [
            "Data flows forward, generating a final output prediction.",
            "We calculate error by comparing output to the goal.",
            "Imagine the loss function as a vast landscape."
        ]
        self.setup_layout("The Forward Pass: Input to Error", lecture_lines_text)
        
        # Assets
        landscape_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/landscape.svg")
        
        # Mobjects
        vec_x = MathTex("X", color=WHITE)
        mat_w = MathTex("W", color=WHITE)
        val_z = MathTex("Z=WX", color="#FF00FF")
        act_y = MathTex(r"Y=\sigma(Z)", color="#00FFFF")
        target_t = MathTex("T", color=WHITE)
        error_e = MathTex(r"E=|Y-T|", color="#FF0000")
        
        # Groups for layout
        vars_group = VGroup(vec_x, mat_w).arrange(RIGHT)
        formula_group = VGroup(val_z, act_y, target_t, error_e).arrange(DOWN)
        
        # Apply layout fixes
        self.place_in_area(vars_group, 'A2', 'A3', scale_factor=0.8)
        self.place_in_area(formula_group, 'B2', 'D3', scale_factor=0.9)
        
        # Visualization of loss landscape
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        surface = axes.plot(lambda x: 0.5 * x**2, color=GRAY)
        axes_surface_group = VGroup(axes, surface, landscape_icon)
        self.place_in_area(axes_surface_group, 'B4', 'F6', scale_factor=0.8)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(vars_group), FadeIn(landscape_icon))
        self.play(Write(val_z))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(Write(act_y))
        self.play(FadeIn(target_t))
        self.play(Write(error_e))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(Create(axes), Create(surface))
        self.wait(2)
