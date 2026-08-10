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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Gradient Descent: Taking the Step", [
            "Final step: Update weights using the calculated gradients.",
            "Iterative learning allows the network to refine itself.",
            "Consistent practice leads to highly accurate predictions."
        ])
        
        # Assets
        MARBLE_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg"
        
        # Create visual elements
        slope = FunctionGraph(lambda x: 0.1 * x**2, x_range=[-3, 3], color=WHITE)
        marble = SVGMobject(MARBLE_PATH).set_color(YELLOW)
        
        # Tangent line placeholder (Yellow)
        tangent_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color=YELLOW)
        
        # Initial state
        pos_ratio = 0.8
        marble.move_to(slope.point_from_proportion(pos_ratio))
        tangent_line.move_to(marble.get_center())
        
        gradient_vec = Arrow(start=marble.get_center(), end=marble.get_center() + RIGHT * 0.4 + DOWN * 0.2, color="#00AAFF")
        loss_text = MathTex("Loss: 0.85", color=WHITE)
        
        # Layout
        self.place_in_area(slope, 'C2', 'F4', scale_factor=0.6)
        self.add(slope, marble, tangent_line, gradient_vec)
        self.place_at_grid(loss_text, 'B6', scale_factor=0.7)
        self.add(loss_text)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED))
        self.play(Indicate(gradient_vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        # Move marble along the surface vector
        new_pos_ratio = 0.5
        new_point = slope.point_from_proportion(new_pos_ratio)
        self.play(
            marble.animate.move_to(new_point),
            tangent_line.animate.move_to(new_point),
            gradient_vec.animate.move_to(new_point)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        loss_text_new = MathTex("Loss: 0.12", color=GREEN)
        self.place_at_grid(loss_text_new, 'B6', scale_factor=0.7)
        self.play(ReplacementTransform(loss_text, loss_text_new))
        self.wait(2)
