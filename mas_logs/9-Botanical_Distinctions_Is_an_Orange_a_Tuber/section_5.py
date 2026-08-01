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
        # 1. Setup layout with the title and lecture lines
        lecture_lines_text = [
            "Let's compare where they grow and their purpose.",
            "Fruits grow in the air for seed dispersal.",
            "Tubers grow in the soil for energy storage.",
            "The orange belongs to the reproduction category.",
            "The potato belongs to the survival storage category."
        ]
        self.setup_layout("Comparison Matrix: Fruit vs. Tuber", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # A coordinate grid (#FFFFFF) appears with axes for 'Location' and 'Function'.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Grid lines for axes
        # X-axis (Location): D2 to D5
        x_axis = Line(self.grid["D2"], self.grid["D5"], color=WHITE)
        # Y-axis (Function): B3 to E3
        y_axis = Line(self.grid["B3"], self.grid["E3"], color=WHITE)
        
        # Axis Titles
        loc_title = Text("Location", font_size=18, color=WHITE)
        # Issue 37: Position loc_title at E5 with scale 0.8
        self.place_at_grid(loc_title, "E5", scale_factor=0.8)
        
        func_title = Text("Function", font_size=18, color=WHITE)
        # Issue 37: Position func_title at B2 with scale 0.8
        self.place_at_grid(func_title, "B2", scale_factor=0.8)
        
        self.play(Create(x_axis), Create(y_axis), Write(loc_title), Write(func_title))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Labels 'Soil' vs 'Air' and 'Storage' vs 'Reproduction' appear on the axes.
        # "Fruits grow in the air for seed dispersal."
        line2_color = BLUE_B
        self.play(self.lecture[1].animate.set_color(line2_color))
        
        air_label = Text("Air", font_size=18, color=line2_color)
        # Issue 38: Position air_label at D5 with scale 0.8
        self.place_at_grid(air_label, "D5", scale_factor=0.8)
        
        repro_label = Text("Reproduction", font_size=18, color=line2_color)
        # Issue 38: Position repro_label at B3 with scale 0.8
        self.place_at_grid(repro_label, "B3", scale_factor=0.8)
        
        self.play(Write(air_label), Write(repro_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Tubers grow in the soil for energy storage."
        line3_color = "#D2B48C" # Tan/Potato color
        self.play(self.lecture[2].animate.set_color(line3_color))
        
        soil_label = Text("Soil", font_size=18, color=line3_color)
        # Scaled to 0.8 for consistency with other labels
        self.place_at_grid(soil_label, "D1", scale_factor=0.8)
        
        storage_label = Text("Storage", font_size=18, color=line3_color)
        # Issue 38: Position storage_label at E3 with scale 0.8
        self.place_at_grid(storage_label, "E3", scale_factor=0.8)
        
        self.play(Write(soil_label), Write(storage_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The Orange icon (#FFA500) moves to the 'Air/Reproduction' quadrant.
        orange_color = "#FFA500"
        self.play(self.lecture[3].animate.set_color(orange_color))
        
        orange_circ = Circle(radius=0.35, color=orange_color, fill_opacity=1.0)
        orange_txt = Text("Orange", font_size=14, color=BLACK)
        orange_icon = VGroup(orange_circ, orange_txt)
        
        # Issue 39: Place orange_icon at B5 (Air/Reproduction quadrant center)
        self.place_at_grid(orange_icon, "B5")
        self.play(FadeIn(orange_icon))
        
        # No additional move needed as it is already at the target quadrant center
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The Potato icon (#D2B48C) moves to the 'Soil/Storage' quadrant.
        # A large 'NOT EQUAL' symbol (#FF0000) appears between the two quadrants.
        potato_color = "#D2B48C"
        self.play(self.lecture[4].animate.set_color(potato_color))
        
        potato_circ = Circle(radius=0.35, color=potato_color, fill_opacity=1.0)
        potato_txt = Text("Potato", font_size=14, color=BLACK)
        potato_icon = VGroup(potato_circ, potato_txt)
        
        # Issue 39: Place potato_icon at E1 initially (Soil/Storage quadrant)
        self.place_at_grid(potato_icon, "E1")
        self.play(FadeIn(potato_icon))
        
        # Move to center of Soil/Storage quadrant (E2)
        self.play(potato_icon.animate.move_to(self.grid["E2"]))
        
        # Not equal symbol at the origin/center (D3)
        neq_symbol = Text("≠", color="#FF0000").scale(2.5)
        self.place_at_grid(neq_symbol, "D3")
        
        self.play(Write(neq_symbol))
        self.wait(2)
