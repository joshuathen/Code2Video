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
        self.setup_layout("Summary & Synthesis", ["SIR models are predictive tools.", "Parameters guide public health policy.", "Summary_Synthesis_Infographic completes our framework."])
        
        # === Animation for Lecture Line 1 ===
        # Display 'S -> I -> R' model diagram
        s_node = Circle(radius=0.3, color=BLUE).add(Text("S", font_size=16))
        i_node = Circle(radius=0.3, color=RED).add(Text("I", font_size=16))
        r_node = Circle(radius=0.3, color=GREEN).add(Text("R", font_size=16))
        
        model_group = VGroup(s_node, i_node, r_node).arrange(RIGHT, buff=0.4)
        arrows = VGroup(
            Arrow(s_node.get_right(), i_node.get_left(), buff=0.05, stroke_width=2, tip_length=0.1),
            Arrow(i_node.get_right(), r_node.get_left(), buff=0.05, stroke_width=2, tip_length=0.1)
        )
        sir_diagram = VGroup(model_group, arrows)
        self.place_at_grid(sir_diagram, 'C3', scale_factor=0.7)
        self.play(Create(sir_diagram))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Flash 'R0' above the model, then fade in 'Intervention'
        r0_label = MathTex(r"R_0", color=YELLOW)
        intervention_label = Text("Intervention", color=ORANGE, font_size=20)
        
        self.place_at_grid(r0_label, 'B3', scale_factor=0.9)
        self.place_at_grid(intervention_label, 'D3', scale_factor=0.8)
        
        self.play(FadeIn(r0_label))
        self.play(Flash(r0_label))
        self.play(FadeIn(intervention_label))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Final screen: 'System Understanding Achieved'
        final_text = Text("System Understanding Achieved", color=WHITE, font_size=24)
        self.place_in_area(final_text, 'E2', 'E5', scale_factor=0.9)
        
        self.play(FadeIn(final_text))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
