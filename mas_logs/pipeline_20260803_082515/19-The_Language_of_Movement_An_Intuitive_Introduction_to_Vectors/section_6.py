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
        # Setup the title and lecture lines
        self.setup_layout("Application: Vectors in the Real World", [
            "Vectors are foundational to modern technology.",
            "Video games use them for movement and physics.",
            "Even AI relies on these simple mathematical arrows."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Vectors are foundational to modern technology.
        # Show the character asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png] 
        # representing a character with a velocity vector.
        self.lecture[0].set_color(WHITE)
        
        # Use ImageMobject for character asset
        character = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        self.place_at_grid(character, "D2", scale_factor=0.6)
        
        vel_start = self.grid["D2"]
        vel_end = self.grid["D4"]
        velocity_arrow = Arrow(vel_start, vel_end, color=WHITE, buff=0)
        vel_label = Text("Velocity", color=WHITE, font_size=16)
        # Issue 42 Fix: Positioned at E4
        self.place_at_grid(vel_label, "E4")
        
        self.play(
            FadeIn(character), 
            Create(velocity_arrow), 
            Write(vel_label), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Video games use them for movement and physics.
        # Add a gold force vector (#FFD700) labeled "Jump" pointing upwards.
        self.lecture[1].set_color("#FFD700")
        
        jump_start = self.grid["D2"]
        jump_end = self.grid["B2"]
        jump_arrow = Arrow(jump_start, jump_end, color="#FFD700", buff=0)
        jump_label = Text("Jump", color="#FFD700", font_size=20)
        # Issue 43 Fix: Positioned at A2
        self.place_at_grid(jump_label, "A2")
        
        self.play(
            Create(jump_arrow), 
            Write(jump_label), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Even AI relies on these simple mathematical arrows.
        # Show the green resulting trajectory vector (#00FF00) as the sum of forces.
        self.lecture[2].set_color("#00FF00")
        
        res_start = self.grid["D2"]
        res_end = self.grid["B4"]
        resultant_arrow = Arrow(res_start, res_end, color="#00FF00", buff=0)
        res_label = Text("Resultant", color="#00FF00", font_size=20)
        self.place_at_grid(res_label, "A4") # Above the resultant arrow
        
        # Visualization of vector addition (parallelogram)
        h_line = DashedLine(jump_end, res_end, color=WHITE, stroke_opacity=0.6)
        v_line = DashedLine(vel_end, res_end, color="#FFD700", stroke_opacity=0.6)
        
        self.play(
            Create(h_line), 
            Create(v_line), 
            run_time=1
        )
        self.play(
            Create(resultant_arrow), 
            Write(res_label), 
            run_time=1.5
        )
        self.wait(3)
