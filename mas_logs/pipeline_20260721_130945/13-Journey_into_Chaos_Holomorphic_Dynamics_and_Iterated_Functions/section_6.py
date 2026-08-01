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
        # Data
        title_text = "Conclusion: Beauty in Complexity"
        lecture_lines = [
            "Simple mathematical rules generate infinite, non-repeating beauty.",
            "These dynamics model everything from weather to fluid flow.",
            "One simple equation contains an entire universe of art."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_BEAUTY = YELLOW
        COLOR_MODEL = "#00FFFF" # Cyan
        COLOR_ART = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Transition from the Mandelbrot set to a montage of natural fractals (leaves, clouds).
        self.play(self.lecture[0].animate.set_color(COLOR_BEAUTY))
        
        # Mandelbrot Set Equation (Representative)
        # Fix 35: Repositioned to A3-A5 to avoid overlap with montage elements.
        mandel_eq = MathTex("z_{n+1} = z_n^2 + c", color=COLOR_BEAUTY)
        self.place_in_area(mandel_eq, 'A3', 'A5', scale_factor=0.8)
        
        self.play(Write(mandel_eq))
        self.wait(1)
        
        # Natural Fractal Elements (Montage)
        # 1. Leaf [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/leaves.svg]
        # Issue 22: Integration of provided leaf asset.
        leaf_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/leaves.svg")
        leaf_icon.set_color(GREEN).set_fill(GREEN, opacity=0.6)
        self.place_at_grid(leaf_icon, 'B3', scale_factor=0.5) 
        leaf_label = Text("Leaves", font_size=18, color=GREEN)
        self.place_at_grid(leaf_label, 'B4', scale_factor=0.8)
        
        # 2. Cloud
        # Issue 36: Positioned at C3/C4 for better grid utilization.
        cloud_icon = VGroup(
            Circle(radius=0.2, color=LIGHT_GREY, fill_opacity=0.7),
            Circle(radius=0.25, color=LIGHT_GREY, fill_opacity=0.7).shift(RIGHT*0.2),
            Circle(radius=0.2, color=LIGHT_GREY, fill_opacity=0.7).shift(LEFT*0.2)
        )
        self.place_at_grid(cloud_icon, 'C3', scale_factor=0.8)
        cloud_label = Text("Clouds", font_size=18, color=LIGHT_GREY)
        self.place_at_grid(cloud_label, 'C4', scale_factor=0.8)
        
        # 3. Crystal
        # Issue 37: Positioned at D3/D4 to prevent crowding bottom edge.
        crystal_icon = RegularPolygon(n=6, color=BLUE_A, fill_opacity=0.5)
        self.place_at_grid(crystal_icon, 'D3', scale_factor=0.7)
        crystal_label = Text("Crystals", font_size=18, color=BLUE_A)
        self.place_at_grid(crystal_label, 'D4', scale_factor=0.8)
        
        # Montage group
        montage_group = VGroup(leaf_icon, leaf_label, cloud_icon, cloud_label, crystal_icon, crystal_label)
        
        # Transition to montage
        self.play(
            FadeIn(montage_group, shift=UP),
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Pulse the entire screen with vibrant, shifting rainbow colors (#FF0000 to #0000FF).
        self.play(self.lecture[1].animate.set_color(COLOR_MODEL))
        
        # Create full-screen rectangle for pulse
        pulse_overlay = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            stroke_width=0,
            fill_opacity=0.0
        ).set_z_index(-1) # Behind everything
        self.add(pulse_overlay)
        
        # Shifting rainbow pulse
        rainbow_colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF"]
        for col in rainbow_colors:
            self.play(
                pulse_overlay.animate.set_fill(color=col, opacity=0.25),
                run_time=0.4,
                rate_func=smooth
            )
        
        self.play(pulse_overlay.animate.set_fill(opacity=0.0), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade everything out to a final 'Thank You' in #FFFFFF.
        self.play(self.lecture[2].animate.set_color(COLOR_ART))
        
        thank_you_text = Text("Thank You", font_size=48, color=WHITE)
        self.place_in_area(thank_you_text, 'B3', 'E5')
        
        # Group current elements (excluding title and lecture)
        current_visuals = VGroup(mandel_eq, montage_group, pulse_overlay)
        
        self.play(
            FadeOut(current_visuals),
            Write(thank_you_text)
        )
        
        self.wait(3)
