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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup Layout
        title = "The Mechanics: Why Light Bends"
        lines = [
            'Imagine a car moving from a floor to carpet.',
            'Approaching at an angle, one wheel hits the carpet.',
            'This wheel slows first, causing the car to pivot.',
            'We measure this angle relative to a perpendicular line.',
            'Light waves pivot similarly when changing speed and materials.'
        ]
        self.setup_layout(title, lines)

        # Assets
        car_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/car.png"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Boundary line between floor and carpet
        boundary_y = (self.grid["C1"][1] + self.grid["D1"][1]) / 2 
        
        floor_rect = Rectangle(width=6.0, height=3.0, fill_color=WHITE, fill_opacity=0.1, stroke_width=0)
        self.place_in_area(floor_rect, "A1", "C6")
        
        carpet_rect = Rectangle(width=6.0, height=3.0, fill_color="#228B22", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(carpet_rect, "D1", "F6")
        
        boundary_line = Line(
            start=[self.grid["C1"][0] - 0.5, boundary_y, 0],
            end=[self.grid["C6"][0] + 0.5, boundary_y, 0],
            color=WHITE, stroke_width=2
        )
        
        # Fix Issue 37: floor_label at B2, scale 0.8
        floor_label = Text("Smooth Floor (Fast)", font_size=16, color=WHITE)
        self.place_at_grid(floor_label, "B2", scale_factor=0.8)
        
        # Fix Issue 36: carpet_label at E2, scale 0.8
        carpet_label = Text("Thick Carpet (Slow)", font_size=16, color="#228B22")
        self.place_at_grid(carpet_label, "E2", scale_factor=0.8)

        self.play(FadeIn(floor_rect), FadeIn(carpet_rect), Create(boundary_line), FadeIn(floor_label), FadeIn(carpet_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.play(self.lecture[1].animate.set_color(BLUE))

        # Issue 26: Load car asset
        car = ImageMobject(car_path)
        car.height = 0.5
        
        # Initial Angle
        initial_angle = -35 * DEGREES
        car.rotate(initial_angle)
        self.place_at_grid(car, "B1")
        
        self.play(FadeIn(car))
        
        # Move car to boundary interface
        target_boundary_pos = self.grid["C3"] + DOWN*0.2
        self.play(car.animate.move_to(target_boundary_pos), run_time=1.5, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Pivot: As front wheel hits, car rotates toward normal (becomes more vertical)
        # Issue 26: Using asset car
        pivot_rot = 20 * DEGREES
        self.play(
            Rotate(car, angle=pivot_rot, about_point=car.get_center()),
            car.animate.shift(DOWN*0.5 + RIGHT*0.2),
            run_time=1.5, rate_func=linear
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Normal line perpendicular to the interface
        normal_x = self.grid["C3"][0]
        normal_line = DashedLine(
            start=[normal_x, self.grid["A3"][1], 0],
            end=[normal_x, self.grid["F3"][1], 0],
            color=WHITE, stroke_width=2
        )
        
        # Fix Issue 35: normal_text at B5, scale 0.7
        normal_text = Text("Normal Line", font_size=16, color=WHITE)
        self.place_at_grid(normal_text, "B5", scale_factor=0.7)
        
        self.play(Create(normal_line), Write(normal_text))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Transition car into wavefronts
        # Issue 26: car [Asset: ...] into light wavefronts (#FFFF00)
        waves = VGroup()
        start_point = self.grid["B3"]
        for i in range(5):
            w = Line(LEFT*0.4, RIGHT*0.4, color=YELLOW, stroke_width=4).rotate(initial_angle)
            dir_vec = np.array([np.cos(initial_angle + PI/2), np.sin(initial_angle + PI/2), 0])
            w.move_to(start_point + dir_vec * (i-2)*0.4)
            waves.add(w)
            
        self.play(FadeOut(car), FadeIn(waves))

        # Bent wavefronts at the boundary
        bent_waves = VGroup()
        for i in range(5):
            ref_x = self.grid["D3"][0] + (i-2)*0.5
            # Top segment (Fast)
            s_top = Line(start=[ref_x - 0.3*np.cos(initial_angle), boundary_y + 0.4, 0],
                        end=[ref_x, boundary_y, 0], color=YELLOW, stroke_width=4)
            # Bottom segment (Slow/Bent)
            s_bot = Line(start=[ref_x, boundary_y, 0],
                        end=[ref_x + 0.3*np.cos(initial_angle + pivot_rot), boundary_y - 0.4, 0], 
                        color=YELLOW, stroke_width=4)
            bent_waves.add(VGroup(s_top, s_bot))
            
        self.play(ReplacementTransform(waves, bent_waves))
        
        self.wait(3)
