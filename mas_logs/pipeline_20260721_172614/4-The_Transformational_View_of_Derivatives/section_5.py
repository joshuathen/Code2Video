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
        # Setup the scene with the specific title and lecture lines for Section 5
        # Removing manual bullets to match storyboard lines exactly
        self.setup_layout("The Formal Transformation Formula", [
            "We can formalize this using the transformation formula.",
            "The change dy equals the derivative times dx.",
            "Here, the derivative acts as a linear map.",
            "It transforms input displacement into output displacement.",
            "This view generalizes easily to higher dimensions."
        ])

        # === Animation for Lecture Line 1 ===
        # Presentation of the formula context
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The change dy (using df in formula for consistency with storyboard)
        self.lecture[1].set_color(WHITE)
        
        # Formula: df = f'(x) * dx (Large White)
        formula1 = MathTex("df", "=", "f'(x)", "\\cdot", "dx", color=WHITE)
        self.place_in_area(formula1, "B2", "B5", scale_factor=1.5)
        self.play(Write(formula1))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The derivative acts as a linear map
        self.lecture[2].set_color(YELLOW)
        
        # Transform the formula into df = 6 * dx using yellow (#FFFF00) for the '6'.
        formula2 = MathTex("df", "=", "6", "\\cdot", "dx")
        formula2[0].set_color(WHITE) # df
        formula2[1].set_color(WHITE) # =
        formula2[2].set_color(YELLOW) # 6
        formula2[3].set_color(WHITE) # \cdot
        formula2[4].set_color(WHITE) # dx
        
        self.place_in_area(formula2, "B2", "B5", scale_factor=1.5)
        
        # Use TransformMatchingTex to keep the transition smooth
        self.play(TransformMatchingTex(formula1, formula2))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Transform input displacement into output displacement
        self.lecture[3].set_color("#FF00FF") # Magenta to match df
        
        # Vector visualization
        # dx vector (Cyan)
        dx_color = "#00FFFF"
        df_color = "#FF00FF"
        
        # Fixed unit length for dx
        dx_unit_length = 0.4
        
        # Create dx group at D3 (Fixing Issue 31)
        dx_vec = Arrow(start=ORIGIN, end=RIGHT * dx_unit_length, color=dx_color, buff=0)
        dx_label = MathTex("dx", color=dx_color, font_size=20)
        dx_group = VGroup(dx_vec, dx_label).arrange(UP, buff=0.1)
        self.place_at_grid(dx_group, "D3", scale_factor=1.0)
        
        # Create initial df group at E3 (Fixing Issue 32)
        # Starts with length dx_unit_length
        df_vec = Arrow(start=ORIGIN, end=RIGHT * dx_unit_length, color=df_color, buff=0)
        df_label = MathTex("df", color=df_color, font_size=20)
        df_group = VGroup(df_vec, df_label).arrange(UP, buff=0.1)
        self.place_at_grid(df_group, "E3", scale_factor=1.0)
        
        self.play(Create(dx_group))
        self.wait(0.5)
        self.play(Create(df_group))
        self.wait(1)
        
        # Scaling df vector to 6x length
        # Using a target group for the transformation
        df_target_vec = Arrow(start=ORIGIN, end=RIGHT * dx_unit_length * 6, color=df_color, buff=0)
        df_target_label = MathTex("df", color=df_color, font_size=20)
        df_target_group = VGroup(df_target_vec, df_target_label).arrange(UP, buff=0.1)
        
        # Position in area E3 to E6 (Fixing Issue 33)
        self.place_in_area(df_target_group, "E3", "E6", scale_factor=1.0)
        
        # Alignment check: making it scale from the same left position as df_group
        df_target_group.move_to(df_group.get_left(), aligned_edge=LEFT)
        
        self.play(Transform(df_group, df_target_group))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.wait(2)
