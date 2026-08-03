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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data setup
        lecture_lines = [
            "Edge detection kernels highlight areas of high contrast.",
            "The Sobel operator detects sharp changes in brightness.",
            "Positive and negative weights reveal boundaries between objects.",
            "Our robot can now see the doorway's sharp outline.",
            "This transforms raw pixels into useful geometric information."
        ]
        self.setup_layout("Real-World Application: Edge Detection", lecture_lines)

        # Assets
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        robot = SVGMobject(robot_path)
        self.place_at_grid(robot, "C1", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Show a 6x6 grid with a vertical edge (left dark, right light) next to a robot.
        self.lecture[0].set_color(YELLOW)
        
        input_grid = VGroup()
        for r in range(6):
            for c in range(6):
                # Create a vertical split: dark grey on left, light grey on right
                color = "#333333" if c < 3 else "#CCCCCC"
                square = Square(side_length=0.4, fill_opacity=1, fill_color=color, stroke_width=1, stroke_color=WHITE)
                input_grid.add(square)
        input_grid.arrange_in_grid(6, 6, buff=0.05)
        self.place_in_area(input_grid, "B2", "E5", scale_factor=1.0)
        
        self.play(FadeIn(input_grid), FadeIn(robot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the Sobel kernel [[-1,0,1],[-2,0,2],[-1,0,1]] in orange #FFA500.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        sobel_values = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
        
        sobel_kernel = VGroup()
        for r in range(3):
            for c in range(3):
                val_text = Text(str(sobel_values[r][c]), font_size=20, color=WHITE)
                cell_bg = Square(side_length=0.5, fill_opacity=0.9, fill_color="#FFA500", stroke_width=2, stroke_color=WHITE)
                cell = VGroup(cell_bg, val_text)
                sobel_kernel.add(cell)
        
        sobel_kernel.arrange_in_grid(3, 3, buff=0.05)
        self.place_at_grid(sobel_kernel, "C6", scale_factor=0.8)
        
        sobel_label = Text("Sobel Kernel", font_size=20, color="#FFA500")
        sobel_label.next_to(sobel_kernel, UP, buff=0.2)
        
        self.play(FadeIn(sobel_kernel), Write(sobel_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slide the kernel across the vertical edge.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Visual indicator of kernel sliding over the input pixels
        sliding_frame = Square(side_length=1.35, color="#FFA500", stroke_width=4, fill_opacity=0.2)
        # Start the frame on the left side (centered on col 1)
        # In a 6x6 grid, index 7 is (row 1, col 1) if 0-indexed and arranged in grid
        sliding_frame.move_to(input_grid[7].get_center()) 
        
        self.play(FadeIn(sliding_frame))
        
        # Slide across the boundary: Col 1 -> Col 2 -> Col 3
        # Boundary is between Col 2 and Col 3
        self.play(sliding_frame.animate.move_to(input_grid[8].get_center()), run_time=1.0)
        self.wait(0.2)
        self.play(sliding_frame.animate.move_to(input_grid[9].get_center()), run_time=1.0)
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show the calculation resulting in a high value at the edge.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # High-level formula representation
        calc_box = VGroup(
            MathTex(r"(-1 \times \text{Dark}) + (1 \times \text{Light})", font_size=24, color=WHITE),
            MathTex(r"= \text{High Positive Value}", font_size=24, color=YELLOW)
        ).arrange(DOWN, buff=0.2)
        self.place_in_area(calc_box, "A2", "A5", scale_factor=1.0)
        
        # Create output grid where edge is highlighted
        output_grid = VGroup()
        for r in range(6):
            for c in range(6):
                # The convolution highlights the boundary (cols 2 and 3)
                is_edge = (c == 2 or c == 3)
                out_color = WHITE if is_edge else BLACK
                square = Square(side_length=0.4, fill_opacity=1, fill_color=out_color, stroke_width=1, stroke_color="#444444")
                output_grid.add(square)
        output_grid.arrange_in_grid(6, 6, buff=0.05)
        self.place_in_area(output_grid, "B2", "E5", scale_factor=1.0)
        
        self.play(Write(calc_box))
        self.play(
            ReplacementTransform(input_grid, output_grid),
            FadeOut(sliding_frame),
            FadeOut(sobel_kernel),
            FadeOut(sobel_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the final 6x6 output showing only the edge line, presented by the robot.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight the detected geometric structure
        highlight_rect = SurroundingRectangle(output_grid, color=WHITE, buff=0.1)
        
        # Robot "presents" it - maybe a small jump or scale
        self.play(
            Create(highlight_rect),
            robot.animate.scale(1.2).set_color(YELLOW),
            run_time=0.5
        )
        self.play(
            Indicate(output_grid, color=YELLOW),
            robot.animate.scale(1/1.2).set_color(WHITE),
            run_time=1
        )
        
        self.wait(2)
