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
        # Setup the layout with section title and lecture lines
        title = "Application: Why It Matters"
        lines = [
            "We can predict outcomes even without knowing everything.",
            "This makes modern quality control and polling possible.",
            "The Central Limit Theorem finds order within chaos."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Show a silhouette of a complex, unknown shape. 
        # Replace it with a predictable Bell Curve. Label 'Predictability' in #FFFFFF.
        self.lecture[0].set_color(YELLOW)
        
        # Complex shape silhouette (irregular polygon)
        complex_shape = Polygon(
            [-0.7, -0.7, 0], [0.7, -0.8, 0], [1.0, -0.2, 0], 
            [0.8, 0.8, 0], [-0.5, 1.0, 0], [-1.0, 0.2, 0],
            color=GRAY, fill_opacity=0.5
        )
        self.place_in_area(complex_shape, "B2", "D5", scale_factor=0.8)
        
        # Bell Curve
        bell_curve = FunctionGraph(
            lambda x: 1.8 * np.exp(-x**2),
            x_range=[-2, 2],
            color=BLUE
        )
        self.place_in_area(bell_curve, "B2", "D5", scale_factor=0.8)
        
        # Label 'Predictability' - FIXED per Issue 32
        predict_label = Text("Predictability", font_size=24, color=WHITE)
        self.place_in_area(predict_label, "E1", "E6", scale_factor=0.8)

        self.play(Create(complex_shape))
        self.wait(1)
        self.play(ReplacementTransform(complex_shape, bell_curve))
        self.play(Write(predict_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Fade in a sequence of cereal box icons on a line. 
        # Label 'Quality Control' in #DA70D6 above them.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(FadeOut(bell_curve), FadeOut(predict_label))
        
        # Sequence of cereal boxes
        boxes = VGroup()
        for i in range(1, 7):
            box_rect = RoundedRectangle(corner_radius=0.1, height=0.6, width=0.4, color=ORANGE, fill_opacity=0.8)
            box_text = Text("Cereal", font_size=8, color=BLACK).move_to(box_rect.get_center())
            box = VGroup(box_rect, box_text)
            self.place_at_grid(box, f"D{i}", scale_factor=1.0)
            boxes.add(box)
            
        # Label 'Quality Control' - FIXED per Issue 31
        qc_label = Text("Quality Control", font_size=24, color="#DA70D6")
        self.place_in_area(qc_label, "B1", "B6", scale_factor=1.0)

        self.play(FadeIn(boxes, shift=UP))
        self.play(Write(qc_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Display 'CLT: Order from Chaos' in the center of the screen in #FFFFFF and scale it up.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(boxes), FadeOut(qc_label))
        
        # Label 'CLT: Order from Chaos' - FIXED per Issue 30
        final_text = Text("CLT: Order from Chaos", font_size=32, color=WHITE)
        self.place_in_area(final_text, "B2", "E5", scale_factor=0.8)
        
        self.play(Write(final_text))
        # Final text size optimized to prevent obstruction; scaling animation removed for clarity.
        self.wait(3)

        # Final cleanup for the section
        self.lecture[2].set_color(WHITE)
        self.wait(1)
