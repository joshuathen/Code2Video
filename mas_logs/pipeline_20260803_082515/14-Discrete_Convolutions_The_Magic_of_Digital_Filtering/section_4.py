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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The Mechanism: Flip, Shift, Multiply, Sum"
        lines = [
            "First, we flip the kernel for mathematical convention.",
            "Next, slide the kernel step-by-step across the signal.",
            "Multiply the overlapping elements of signal and kernel.",
            "Sum these products to find the output value.",
            "Repeat this process for every position in the signal."
        ]
        self.setup_layout(title, lines)

        # Colors
        SIGNAL_COLOR = BLUE
        KERNEL_COLOR = ORANGE
        FLIPPED_COLOR = "#FF69B4"
        SUM_COLOR = YELLOW
        OUTPUT_COLOR = GREEN

        # Helper to create array mobjects
        def create_array(values, color=WHITE, cell_size=0.8):
            cells = VGroup()
            for val in values:
                square = Square(side_length=cell_size, color=color)
                label = Text(str(val), font_size=24, color=color)
                cell = VGroup(square, label)
                cells.add(cell)
            cells.arrange(RIGHT, buff=0.1)
            return cells

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(FLIPPED_COLOR)
        
        signal_vals = [0, 1, 2, 1, 0]
        kernel_vals = [1, 0, -1]
        flipped_vals = [-1, 0, 1]
        
        signal_array = create_array(signal_vals, color=SIGNAL_COLOR)
        self.place_in_area(signal_array, "A1", "A5", scale_factor=0.8)
        signal_label = Text("Signal", font_size=20, color=SIGNAL_COLOR).next_to(signal_array, UP, buff=0.2)
        
        kernel_array = create_array(kernel_vals, color=KERNEL_COLOR)
        # Fix Issue 31: Move from C1-C3 to C2-C4
        self.place_in_area(kernel_array, "C2", "C4", scale_factor=0.8)
        kernel_label = Text("Kernel", font_size=20, color=KERNEL_COLOR).next_to(kernel_array, UP, buff=0.2)
        
        self.play(FadeIn(signal_array), FadeIn(signal_label))
        self.play(FadeIn(kernel_array), FadeIn(kernel_label))
        self.wait(1)
        
        # Flip Animation
        flipped_kernel = create_array(flipped_vals, color=FLIPPED_COLOR)
        # Fix Issue 31: Move from C1-C3 to C2-C4
        self.place_in_area(flipped_kernel, "C2", "C4", scale_factor=0.8)
        flipped_label = Text("Flipped Kernel", font_size=20, color=FLIPPED_COLOR).next_to(flipped_kernel, UP, buff=0.2)
        
        self.play(
            ReplacementTransform(kernel_array, flipped_kernel),
            ReplacementTransform(kernel_label, flipped_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(FLIPPED_COLOR)
        
        # Position 1: Kernel center under signal[1]
        pos1 = signal_array[1].get_center() + DOWN * 1.2
        self.play(flipped_kernel.animate.move_to(pos1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(SUM_COLOR)
        
        # Show multiplications for pos 1
        m_labels = VGroup()
        for i in range(3):
            m_label = Text(f"{signal_vals[i]}×{flipped_vals[i]}", font_size=18, color=WHITE)
            m_label.move_to(flipped_kernel[i].get_center() + DOWN * 0.7)
            m_labels.add(m_label)
        
        self.play(FadeIn(m_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SUM_COLOR)
        
        sum_val_1 = sum([signal_vals[i]*flipped_vals[i] for i in range(3)]) 
        sum_text = Text(f"Sum = {sum_val_1}", font_size=24, color=SUM_COLOR)
        # Fix Issue 32: Move from E2 to E3
        self.place_at_grid(sum_text, "E3")
        
        output_array = create_array(["?", "?", "?"], color=OUTPUT_COLOR)
        # Fix Issue 33: Move from F1-F3 to F2-F4
        self.place_in_area(output_array, "F2", "F4", scale_factor=0.8)
        output_label = Text("Output", font_size=20, color=OUTPUT_COLOR).next_to(output_array, UP, buff=0.2)
        
        self.play(Write(sum_text), FadeIn(output_array), FadeIn(output_label))
        
        out_val_1 = Text(str(sum_val_1), font_size=24, color=OUTPUT_COLOR).move_to(output_array[0][1].get_center())
        self.play(
            ReplacementTransform(sum_text, out_val_1),
            FadeOut(output_array[0][1])
        )
        output_array[0].remove(output_array[0][1])
        output_array[0].add(out_val_1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(OUTPUT_COLOR)
        
        # Second Position
        pos2 = signal_array[2].get_center() + DOWN * 1.2
        sum_val_2 = sum([signal_vals[i+1]*flipped_vals[i] for i in range(3)])
        
        self.play(
            flipped_kernel.animate.move_to(pos2),
            FadeOut(m_labels)
        )
        
        m_labels2 = VGroup()
        for i in range(3):
            m_label = Text(f"{signal_vals[i+1]}×{flipped_vals[i]}", font_size=18, color=WHITE)
            m_label.move_to(flipped_kernel[i].get_center() + DOWN * 0.7)
            m_labels2.add(m_label)
            
        self.play(FadeIn(m_labels2))
        sum_text2 = Text(f"Sum = {sum_val_2}", font_size=24, color=SUM_COLOR)
        # Fix Issue 32: Move from E2 to E3
        self.place_at_grid(sum_text2, "E3")
        self.play(Write(sum_text2))
        
        out_val_2 = Text(str(sum_val_2), font_size=24, color=OUTPUT_COLOR).move_to(output_array[1][1].get_center())
        self.play(
            ReplacementTransform(sum_text2, out_val_2),
            FadeOut(output_array[1][1])
        )
        output_array[1].remove(output_array[1][1])
        output_array[1].add(out_val_2)
        
        # Third Position
        pos3 = signal_array[3].get_center() + DOWN * 1.2
        sum_val_3 = sum([signal_vals[i+2]*flipped_vals[i] for i in range(3)])
        
        self.play(
            flipped_kernel.animate.move_to(pos3),
            FadeOut(m_labels2)
        )
        
        m_labels3 = VGroup()
        for i in range(3):
            m_label = Text(f"{signal_vals[i+2]}×{flipped_vals[i]}", font_size=18, color=WHITE)
            m_label.move_to(flipped_kernel[i].get_center() + DOWN * 0.7)
            m_labels3.add(m_label)
            
        self.play(FadeIn(m_labels3))
        sum_text3 = Text(f"Sum = {sum_val_3}", font_size=24, color=SUM_COLOR)
        # Fix Issue 32: Move from E2 to E3
        self.place_at_grid(sum_text3, "E3")
        self.play(Write(sum_text3))
        
        out_val_3 = Text(str(sum_val_3), font_size=24, color=OUTPUT_COLOR).move_to(output_array[2][1].get_center())
        self.play(
            ReplacementTransform(sum_text3, out_val_3),
            FadeOut(output_array[2][1])
        )
        output_array[2].remove(output_array[2][1])
        output_array[2].add(out_val_3)
        
        self.play(FadeOut(m_labels3))
        self.wait(2)
