from manim import *
import numpy as np

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
        lecture_lines = [
            "Measuring a system forces it to pick one state.",
            "The vector instantly snaps to a single classical axis.",
            "This sudden change is known as 'Wavefunction Collapse'.",
            "Probability depends on the square of the state's amplitude.",
            "The mystery of 'both' vanishes upon our observation."
        ]
        self.setup_layout("The Collapse: The Act of Measuring", lecture_lines)

        # Colors
        COLOR_PSI = "#FF69B4"   # Pink from section 3
        COLOR_ZERO = "#FFFF00"  # Yellow for |0> (Y-axis)
        COLOR_ALPHA = "#00FF00" # Green
        COLOR_BETA = "#FF0000"  # Red
        COLOR_CAMERA = "#CCCCCC" # Grey
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Coordinate system - Adjusted per Issue 44
        axes = Axes(
            x_range=[0, 1.5, 1],
            y_range=[0, 1.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, 'B2', 'F5')
        
        label_0 = MathTex("|0\\rangle", color=COLOR_ZERO).scale(0.8)
        label_0.next_to(axes.c2p(0, 1), LEFT, buff=0.1)
        
        label_1 = MathTex("|1\\rangle", color=WHITE).scale(0.8)
        label_1.next_to(axes.c2p(1, 0), DOWN, buff=0.1)

        # Initial Vector |psi> at 45 degrees
        vec_end_init = axes.c2p(np.sqrt(0.5), np.sqrt(0.5))
        vec_psi = Arrow(
            start=axes.c2p(0, 0),
            end=vec_end_init,
            buff=0,
            color=COLOR_PSI
        )
        label_psi = MathTex("|\\psi\\rangle", color=COLOR_PSI).scale(0.8)
        label_psi.next_to(vec_psi.get_end(), UR, buff=0.1)

        # Projections
        proj_alpha = DashedLine(
            axes.c2p(np.sqrt(0.5), 0), 
            axes.c2p(np.sqrt(0.5), np.sqrt(0.5)), 
            color=COLOR_ALPHA
        )
        proj_beta = DashedLine(
            axes.c2p(0, np.sqrt(0.5)), 
            axes.c2p(np.sqrt(0.5), np.sqrt(0.5)), 
            color=COLOR_BETA
        )
        
        label_alpha = MathTex("\\alpha", color=COLOR_ALPHA).scale(0.8)
        label_alpha.next_to(axes.c2p(np.sqrt(0.5), 0), DOWN, buff=0.1)
        
        label_beta = MathTex("\\beta", color=COLOR_BETA).scale(0.8)
        label_beta.next_to(axes.c2p(0, np.sqrt(0.5)), LEFT, buff=0.1)

        # Camera Icon - Updated per Issue 32 & 45
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        camera.set_color(COLOR_CAMERA)
        self.place_at_grid(camera, 'B6', scale_factor=0.8)

        self.add(axes, label_0, label_1)
        self.play(Create(vec_psi), Write(label_psi))
        self.play(Create(proj_alpha), Create(proj_beta), Write(label_alpha), Write(label_beta))
        self.play(FadeIn(camera))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)

        # Preparation for snap: ghosts
        ghost_psi = vec_psi.copy().set_stroke(opacity=0.3)
        ghost_proj_alpha = proj_alpha.copy().set_stroke(opacity=0.3)
        ghost_label_alpha = label_alpha.copy().set_opacity(0.3)

        # Flash effect - Asset usage updated per Issue 32 (camera flashes)
        flash = Circle(radius=0.1, color=WHITE, fill_opacity=0.9).move_to(camera.get_center())
        self.play(flash.animate.scale(50).set_opacity(0), run_time=0.4)
        self.remove(flash)
        
        # Add ghosts to scene
        self.add(ghost_psi, ghost_proj_alpha, ghost_label_alpha)

        # Vector snaps to Y-axis |0>
        target_end = axes.c2p(0, 1)
        self.play(
            vec_psi.animate.put_start_and_end_on(axes.c2p(0,0), target_end).set_color(COLOR_ZERO),
            label_psi.animate.next_to(target_end, UP, buff=0.1).set_color(COLOR_ZERO),
            proj_beta.animate.set_stroke(opacity=0),
            proj_alpha.animate.set_stroke(opacity=0),
            label_beta.animate.set_color(COLOR_ZERO),
            run_time=0.2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Emphasize collapse
        self.play(Indicate(vec_psi, color=COLOR_ZERO, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)

        # Show probability relationship - Updated per Issue 46
        prob_math = MathTex("P(|0\\rangle) = |\\beta|^2", color=COLOR_ZERO)
        self.place_at_grid(prob_math, 'D6', scale_factor=1.2)
        
        self.play(Write(prob_math))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)

        # Fade out ghosts and extra indicators
        self.play(
            FadeOut(ghost_psi),
            FadeOut(ghost_proj_alpha),
            FadeOut(ghost_label_alpha),
            FadeOut(camera),
            FadeOut(label_alpha),
            FadeOut(label_beta),
            prob_math.animate.set_color(WHITE),
            run_time=1.5
        )
        self.wait(2)
