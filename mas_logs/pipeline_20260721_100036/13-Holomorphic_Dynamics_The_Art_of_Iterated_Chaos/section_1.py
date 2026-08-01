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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_str = "The Canvas: The Complex Plane and the Feedback Loop"
        lecture_lines = [
            "Complex numbers represent points on a 2D plane.",
            "Iteration creates a mathematical feedback loop.",
            "The output of one step becomes the next input.",
            "Points jump across the plane following this rule.",
            "Holomorphic Dynamics studies these paths over time."
        ]
        
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Complex numbers represent points on a 2D plane.
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-2, 2, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"color": BLUE_C, "include_tip": True}
        )
        # Apply VideoCritic Fix: Area A2-F6, scale 0.9 (Issue 20)
        self.place_in_area(axes, "A2", "F6", scale_factor=0.9)
        
        # Initial point 'Pixel' using Asset (Issue 16)
        pixel_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg"
        pixel = SVGMobject(pixel_asset_path, height=0.4)
        z0_pos = axes.c2p(1, 0.5)
        pixel.move_to(z0_pos)
        
        pixel_label = MathTex("z_0 = a + bi", font_size=24, color=WHITE).next_to(pixel, UR, buff=0.1)
        
        self.play(Create(axes), run_time=1)
        self.play(FadeIn(pixel), Write(pixel_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Iteration creates a mathematical feedback loop.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        f_box = Rectangle(width=1.2, height=0.8, color=WHITE, fill_opacity=0.2)
        # Apply VideoCritic Fix: Grid A6, scale 0.8 (Issue 19)
        self.place_at_grid(f_box, "A6", scale_factor=0.8)
        f_label = MathTex("f(z)", font_size=24, color=WHITE).move_to(f_box)
        
        # Visualizing the feedback loop
        arrow_in = CurvedArrow(pixel.get_top(), f_box.get_left(), angle=-PI/3, color=GRAY_B)
        arrow_out = CurvedArrow(f_box.get_bottom(), pixel.get_bottom(), angle=PI/2, color=GRAY_B)
        
        self.play(Create(f_box), Write(f_label))
        self.play(Create(arrow_in), Create(arrow_out))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The output of one step becomes the next input.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Jump sequence
        z_coords = [(1, 0.5), (2, 0.8), (3, -0.5), (0.5, -1.2)]
        ghost_dots = VGroup()
        orbit_labels = VGroup(pixel_label)
        
        self.play(FadeOut(arrow_in), FadeOut(arrow_out))

        for i in range(1, 4):
            target_pos = axes.c2p(z_coords[i][0], z_coords[i][1])
            new_label = MathTex(f"z_{i}", font_size=20, color=WHITE).next_to(target_pos, DR, buff=0.1)
            
            # Leave a dot at the previous position
            ghost = Dot(pixel.get_center(), color=WHITE, radius=0.04, fill_opacity=0.4)
            self.add(ghost)
            ghost_dots.add(ghost)

            # Jump animation using the pixel asset (Issue 16)
            self.play(
                Indicate(f_box, color=YELLOW),
                pixel.animate(path_arc=PI/3).move_to(target_pos),
                run_time=0.8
            )
            orbit_labels.add(new_label)
            self.play(FadeIn(new_label), run_time=0.4)

        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Points jump across the plane following this rule.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        
        # Add final ghost dot for z3
        ghost_final = Dot(pixel.get_center(), color=WHITE, radius=0.04, fill_opacity=0.4)
        self.add(ghost_final)
        ghost_dots.add(ghost_final)

        # Build the orbit path
        orbit_points = [axes.c2p(x, y) for x, y in z_coords]
        orbit_path = VMobject(color="#FFFF00")
        orbit_path.set_points_as_corners(orbit_points)
        
        self.play(Create(orbit_path), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Holomorphic Dynamics studies these paths over time.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # Clean up the plane to focus on the orbit
        self.play(
            FadeOut(axes),
            FadeOut(f_box),
            FadeOut(f_label),
            FadeOut(orbit_labels),
            FadeOut(ghost_dots),
            FadeOut(pixel)
        )
        # Final emphasis on the orbit
        self.play(orbit_path.animate.scale(1.5).move_to(self.grid["C4"]), run_time=1.5)
        self.wait(2)
