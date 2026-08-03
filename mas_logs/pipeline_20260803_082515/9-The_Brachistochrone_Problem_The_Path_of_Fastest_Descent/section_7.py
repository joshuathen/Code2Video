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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize Scene Layout
        self.setup_layout("Summary and Real-World Application", [
            "Optimization requires balancing speed and distance.",
            "Cycloids appear in roller coasters and nature.",
            "The fastest path isn't always the straightest one."
        ])

        # Colors defined in requirements
        COASTER_COLOR = "#00BFFF"
        FALCON_COLOR = "#FFD700"
        SUMMARY_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "Optimization requires balancing speed and distance."
        self.lecture[0].set_color(YELLOW)

        # Split screen setup
        coaster_label = Text("Roller Coaster", font_size=20, color=COASTER_COLOR)
        self.place_at_grid(coaster_label, "A2", scale_factor=0.8)
        
        falcon_label = Text("Falcon Dive", font_size=20, color=FALCON_COLOR)
        self.place_at_grid(falcon_label, "A5", scale_factor=0.8)

        # Static representations for context
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg]
        coaster_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        coaster_icon.set_color(COASTER_COLOR)
        self.place_in_area(coaster_icon, "B1", "C3", scale_factor=0.6)
        
        falcon_icon = Triangle(color=FALCON_COLOR).rotate(-PI/2)
        # Fix for Issue 42: Adjust scale factor to 0.5
        self.place_in_area(falcon_icon, "B4", "C6", scale_factor=0.5)

        self.play(
            FadeIn(coaster_label),
            FadeIn(falcon_label),
            FadeIn(coaster_icon),
            FadeIn(falcon_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Cycloids appear in roller coasters and nature."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Cycloid Path Generation (Brachistochrone shape)
        def create_cycloid():
            return ParametricFunction(
                lambda t: np.array([t - np.sin(t), -(1 - np.cos(t)), 0]),
                t_range=[0, PI],
                color=WHITE
            )

        cycloid_path_1 = create_cycloid()
        # Fix for Issue 41: Adjust scale factor to 0.8
        self.place_in_area(cycloid_path_1, "B1", "C3", scale_factor=0.8)
        
        cycloid_path_2 = create_cycloid()
        # Fix for Issue 41: Adjust scale factor to 0.8
        self.place_in_area(cycloid_path_2, "B4", "C6", scale_factor=0.8)

        # Trace the curves
        self.play(
            Create(cycloid_path_1),
            Create(cycloid_path_2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The fastest path isn't always the straightest one."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        summary_text = Text("Optimization = Balance of\nDistance and Speed", font_size=28, color=SUMMARY_COLOR)
        # Fix for Issue 40: Adjust positioning and scale
        self.place_in_area(summary_text, 'E1', 'F6', scale_factor=0.6)

        # Display summary in the center area while dimming previous visuals
        self.play(
            coaster_label.animate.set_fill(opacity=0.2),
            falcon_label.animate.set_fill(opacity=0.2),
            coaster_icon.animate.set_opacity(0.2),
            falcon_icon.animate.set_fill(opacity=0.2),
            cycloid_path_1.animate.set_stroke(opacity=0.2),
            cycloid_path_2.animate.set_stroke(opacity=0.2),
            Write(summary_text)
        )
        self.wait(3)

        # Cleanup
        self.play(
            *[FadeOut(m) for m in self.mobjects]
        )
