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

class Section1Scene(TeachingScene):
    def construct(self):
        # Finalized lecture lines from ScriptWriter
        lecture_lines = [
            "Meet Cargo, a robot powered by two batteries.",
            "Each battery has a random, uncertain lifetime.",
            "How long can Cargo travel before stopping?",
            "We define total time Z as X plus Y.",
            "To find Z, we must merge their distributions."
        ]
        
        self.setup_layout("The Quest: Combining Uncertainties", lecture_lines)
        
        # Colors
        color_a = "#00FF00" # Green
        color_b = "#FFFF00" # Yellow
        color_z = "#00FFFF" # Cyan
        color_main = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        # Fade in 'Cargo' robot (#FFFFFF) and two batteries (#00FF00, #FFFF00).
        # Building a simple robot representation
        cargo_body = RoundedRectangle(corner_radius=0.1, width=0.8, height=1.0, color=color_main)
        cargo_head = Square(side_length=0.3, color=color_main).next_to(cargo_body, UP, buff=0.1)
        cargo_label = Text("Cargo", font_size=14).move_to(cargo_body.get_center())
        cargo = VGroup(cargo_body, cargo_head, cargo_label)
        
        battery_a = Rectangle(width=0.3, height=0.5, color=color_a, fill_opacity=0.5)
        battery_a_cap = Rectangle(width=0.15, height=0.08, color=color_a, fill_opacity=1).next_to(battery_a, UP, buff=0)
        bat_a_grp = VGroup(battery_a, battery_a_cap)
        bat_a_txt = MathTex("X", color=color_a, font_size=20).next_to(bat_a_grp, DOWN, buff=0.1)
        
        battery_b = Rectangle(width=0.3, height=0.5, color=color_b, fill_opacity=0.5)
        battery_b_cap = Rectangle(width=0.15, height=0.08, color=color_b, fill_opacity=1).next_to(battery_b, UP, buff=0)
        bat_b_grp = VGroup(battery_b, battery_b_cap)
        bat_b_txt = MathTex("Y", color=color_b, font_size=20).next_to(bat_b_grp, DOWN, buff=0.1)

        # Apply Fix for Issue 28: Scale factor 1.0 for cargo at B3
        self.place_at_grid(cargo, "B3", scale_factor=1.0)
        self.place_at_grid(bat_a_grp, "B2", scale_factor=0.8)
        self.place_at_grid(bat_b_grp, "B4", scale_factor=0.8)
        bat_a_txt.next_to(bat_a_grp, DOWN, buff=0.1)
        bat_b_txt.next_to(bat_b_grp, DOWN, buff=0.1)

        self.lecture[0].set_color(color_main)
        self.play(
            FadeIn(cargo),
            FadeIn(bat_a_grp, bat_a_txt),
            FadeIn(bat_b_grp, bat_b_txt),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display PDF curve for Battery A (X) (#00FF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_a)
        
        axes_a = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.2, 0.4], 
            x_length=2.2, y_length=1.5, 
            axis_config={"include_tip": False, "font_size": 16}
        ).set_color(GREY)
        pdf_a = axes_a.plot(lambda x: 1.0 * np.exp(-(x-2)**2 / 0.5), color=color_a)
        label_a = MathTex("f_X(x)", color=color_a, font_size=24).next_to(axes_a, UP, buff=0.1)
        pdf_a_grp = VGroup(axes_a, pdf_a, label_a)
        
        self.place_in_area(pdf_a_grp, "C2", "D3", scale_factor=0.85)
        
        self.play(Create(axes_a), Create(pdf_a), Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display PDF curve for Battery B (Y) (#FFFF00).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_b)
        
        axes_b = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.2, 0.4], 
            x_length=2.2, y_length=1.5, 
            axis_config={"include_tip": False, "font_size": 16}
        ).set_color(GREY)
        pdf_b = axes_b.plot(lambda x: 1.0 * np.exp(-(x-1.5)**2 / 0.8), color=color_b)
        label_b = MathTex("f_Y(y)", color=color_b, font_size=24).next_to(axes_b, UP, buff=0.1)
        pdf_b_grp = VGroup(axes_b, pdf_b, label_b)
        
        self.place_in_area(pdf_b_grp, "C4", "D5", scale_factor=0.85)
        
        self.play(Create(axes_b), Create(pdf_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Write equation Z = X + Y in center screen (#FFFFFF).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(color_main)
        
        eq_z = MathTex("Z = X + Y", color=color_main, font_size=32)
        # Apply Fix for Issue 27: Move eq_z to B5 to avoid overlap
        self.place_at_grid(eq_z, "B5", scale_factor=1.0)
        
        self.play(Write(eq_z))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Merge X and Y curves into a new distribution Z (#00FFFF).
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(color_z)
        
        axes_z = Axes(
            x_range=[0, 8, 2], y_range=[0, 0.8, 0.2], 
            x_length=4.0, y_length=1.8,
            axis_config={"include_tip": False, "font_size": 16}
        ).set_color(GREY)
        # Sum of Gaussian-like curves: Mean 2+1.5=3.5. 
        pdf_z = axes_z.plot(lambda x: 0.6 * np.exp(-(x-3.5)**2 / 1.5), color=color_z)
        label_z = MathTex("f_Z(z)", color=color_z, font_size=24).next_to(axes_z, UP, buff=0.1)
        pdf_z_grp = VGroup(axes_z, pdf_z, label_z)
        
        # Apply Fix for Issue 26: Use larger area C2-F5 and scale 0.8 to prevent compression
        self.place_in_area(pdf_z_grp, "C2", "F5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(VGroup(pdf_a_grp, pdf_b_grp), pdf_z_grp),
            FadeOut(eq_z),
            FadeOut(cargo),
            FadeOut(bat_a_grp, bat_a_txt),
            FadeOut(bat_b_grp, bat_b_txt),
            run_time=1.5
        )
        self.wait(2)
        self.lecture[4].set_color(WHITE)
